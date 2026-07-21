"""
RBM a cascata per un processo di diffusione discreto (stile binary/Bernoulli
diffusion, cfr. Sohl-Dickstein et al.).  Una RBM indipendente per ogni step t,
addestrata con CD-k.

Startup commands:
python rbm_diffusion.py --exp_id run01 --train
python rbm_diffusion.py --exp_id run01 --generate (deprecated)
python rbm_diffusion.py --exp_id run01 --multigenerate

Carlo aironi, 2026
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.datasets import fetch_openml
from tqdm import tqdm

# Hyperparameters
T = 8             # numero di passi di diffusione (= numero di RBM)
NV = 28 * 28      # unita' visibili (pixel)
NH = 256          # unita' latenti per RBM
EPOCHS = 25
BATCH = 128
K_TRAIN = 5       # Gibbs steps per la fase negativa (CD-k)
K_GEN = 50        # Gibbs steps per la generazione
LR = 0.01
DATASET = "Fashion MNIST"    # "MNIST digit" | "Fashion MNIST"
NOISE_KIND = "linear"        # "linear" | "quadratic" | "sigmoid" | "cosine" | "geometric"
SEED = 0                     # seed base, ogni RBM usa SEED + t


# Data loading
def load_dataset(name: str, data_home: str | None = None):
    if name == "MNIST digit":
        print("Download MNIST digit dataset...")
        dset = fetch_openml("mnist_784", version=1, as_frame=False, data_home=data_home)
    elif name == "Fashion MNIST":
        print("Download Fashion MNIST dataset...")
        dset = fetch_openml("Fashion-MNIST", version=1, as_frame=False, data_home=data_home)
    else:
        raise ValueError(f"Dataset sconosciuto: {name}")

    X = dset.data.astype(np.float32) / 255.0
    y = dset.target
    # binarization: threshold 0.3, poi {0,1} -> {-1,+1}
    X_bin = (X > 0.3).astype(np.float32) * 2 - 1
    n_samples, n_visible = X_bin.shape
    side = int(np.sqrt(n_visible))
    print(f"n_samples = {n_samples}, n_visible = {n_visible}  (immagini {side}x{side})")
    return X_bin, y


def _p_to_gamma(p):
    """Campo esterno associato alla probabilita' di flip (formula logit)."""
    return np.where(p < 0.5, 0.5 * np.log((1 - p) / p), 0.0)


def _cumulative_to_stepwise(alpha_bar):
    alpha_prev = np.concatenate(([1.0], alpha_bar[:-1]))
    r = alpha_bar / alpha_prev
    r = np.clip(r, 1e-6, 1.0)          # evita p_t < 0 per rumore numerico
    p = (1 - r) / 2
    return np.clip(p, 1e-6, 0.5)


def noise_schedule(T: int, p_min: float = 0.1, p_max: float = 0.5, kind: str = "linear", s: float = 0.008):
    """Restituisce (p, gamma) per T step di diffusione, secondo il profilo scelto.

    kind:
      - "linear"    : p_t lineare in [p_min, p_max] (schedule originale)
      - "quadratic" : p_t cresce con t^2, corruzione lenta all'inizio, rapida alla fine
      - "sigmoid"   : transizione a S centrata a meta' schedule (utile per T grandi)
      - "cosine"    : profilo cumulato coseno (Nichol & Dhariwal / D3PM), poi invertito
                      in p_t step-wise. p_max qui e' ignorato: il tetto e' implicito nel coseno.
      - "geometric" : retention (1 - 2p_t) costante per step -> decadimento
                      esponenziale (log-lineare) del segnale cumulato, equivalente
                      a SNR che scende linearmente in scala log.
    s: offset di piccola entita' per il profilo "cosine" (come in Nichol & Dhariwal, s=0.008).
    """
    t = np.arange(1, T + 1)

    if kind == "linear":
        p = np.linspace(p_min, p_max, T)

    elif kind == "quadratic":
        frac = np.linspace(0, 1, T) ** 2
        p = p_min + (p_max - p_min) * frac

    elif kind == "sigmoid":
        x = np.linspace(-6, 6, T)
        frac = sigmoid(x)
        p = p_min + (p_max - p_min) * frac

    elif kind == "cosine":
        # alpha_bar_t = f(t)/f(0), f(t) = cos^2( ((t/T + s)/(1+s)) * pi/2 )
        f = lambda tt: np.cos(((tt / T + s) / (1 + s)) * np.pi / 2) ** 2
        alpha_bar = f(t) / f(0)
        alpha_bar = np.clip(alpha_bar, 1e-6, 1.0)
        p = _cumulative_to_stepwise(alpha_bar)

    elif kind == "geometric":
        # retention (1 - 2p) costante per step: (1-2p)^T = target di retention cumulata finale.
        # p_max e' il flip "finale" desiderato ma va tenuto strettamente < 0.5, altrimenti
        # la retention cumulata target e' 0 e il rate per-step collassa (p_t=0.5 gia' al primo step).
        p_max_eff = min(p_max, 0.499)
        r = (1 - 2 * p_max_eff) ** (1.0 / T)
        p = np.full(T, (1 - r) / 2)

    else:
        raise ValueError(f"kind sconosciuto: {kind}")

    p = np.clip(p, 1e-6, 0.5)
    gamma = _p_to_gamma(p)
    return p, gamma


# RBM
class RBM:
    """
    RBM {-1,+1} associata a un singolo step t del processo di diffusione.
    """

    def __init__(self, nv, nh, t, p_schedule, gamma_schedule, sigma=0.01, seed=None):
        self.nv, self.nh, self.t = nv, nh, t
        self.p = p_schedule
        self.gamma = gamma_schedule
        self.rng = np.random.default_rng(seed)

        self.W = (sigma * self.rng.standard_normal((nv, nh))).astype(np.float32)
        self.b = np.zeros(nv, dtype=np.float32)
        self.c = np.zeros(nh, dtype=np.float32)

    def h_prob(self, v):
        return sigmoid(2 * (self.c + v @ self.W))

    def v_prob(self, h, field):
        return sigmoid(2 * (self.b + field + h @ self.W.T))

    def flip(self, x, prob):
        return np.where(self.rng.random(x.shape) < prob, -x, x)

    def q_sample_pair(self, x0):
        """
        Campiona (x^{t-1}, x^t) dal processo forward per lo step self.t.
        """
        x = x0
        for s in range(self.t - 1):
            x = self.flip(x, self.p[s])
        x_prev = x
        x_t = self.flip(x_prev, self.p[self.t - 1])
        return x_prev, x_t

    def gibbs(self, v, field, k):
        for _ in range(k):
            h_p = self.h_prob(v)
            h = np.where(self.rng.random(h_p.shape) < h_p, 1.0, -1.0).astype(np.float32)
            v_p = self.v_prob(h, field)
            v = np.where(self.rng.random(v_p.shape) < v_p, 1.0, -1.0).astype(np.float32)
        return v, v_p

    def contrastive_divergence(self, x0_batch, k, lr):
        """
        Un passo di CD-k su un batch; ritorna l'errore di ricostruzione.
        """
        x_prev, x_t = self.q_sample_pair(x0_batch)
        field = self.gamma[self.t - 1] * x_t

        hp = self.h_prob(x_prev)                        # fase positiva
        v, _ = self.gibbs(x_prev.copy(), field, k)      # fase negativa
        hn = self.h_prob(v)

        m = x0_batch.shape[0]
        # implementazione mean-field
        self.W += lr * (x_prev.T @ hp - v.T @ hn) / m
        self.b += lr * (x_prev.mean(0) - v.mean(0))
        self.c += lr * (hp.mean(0) - hn.mean(0))
        
        p_v_recon = self.v_prob(hp, field)
        x_prev_01 = (x_prev + 1) / 2
        return float(np.mean((x_prev_01 - p_v_recon) ** 2))

    def train(self, X0, epochs, batch_size, k, lr, verbose=True):
        n = X0.shape[0]
        recon_errors = []
        for ep in range(epochs):
            if verbose:
                print(f"Timestep {self.t} (RBM#{self.t})")
            perm = self.rng.permutation(n)
            iterator = range(0, n, batch_size)
            if verbose:
                iterator = tqdm(iterator, desc=f"EPOCH {ep + 1}/{epochs}")
            for i in iterator:
                x0_batch = X0[perm[i:i + batch_size]]
                recon_errors.append(self.contrastive_divergence(x0_batch, k, lr))
        return recon_errors

    def save(self, path):
        """
        Salva pesi + metadati (t, nv, nh) in un unico file .npz.
        """
        np.savez(path, W=self.W, b=self.b, c=self.c,
                  t=self.t, nv=self.nv, nh=self.nh)

    @classmethod
    def load(cls, path, p_schedule, gamma_schedule, seed=None):
        data = np.load(path)
        rbm = cls(int(data["nv"]), int(data["nh"]), int(data["t"]),
                   p_schedule, gamma_schedule, seed=seed)
        rbm.W, rbm.b, rbm.c = data["W"], data["b"], data["c"]
        return rbm

# Persistenza dell'intera cascata (T RBM)
def save_rbms(rbms, directory):
    """
    rbms: dict {t: RBM} oppure list indicizzata 0..T-1 (t = indice+1).
    """
    os.makedirs(directory, exist_ok=True)
    items = rbms.items() if isinstance(rbms, dict) else enumerate(rbms, start=1)
    for t, rbm in items:
        rbm.save(os.path.join(directory, f"rbm_t{t:02d}.npz"))


def load_rbms(directory, T, p, gamma, base_seed=0):
    """
    Ricarica le T RBM salvate con save_rbms. Ritorna un dict {t: RBM}.
    """
    rbms = {}
    for t in range(1, T + 1):
        path = os.path.join(directory, f"rbm_t{t:02d}.npz")
        rbms[t] = RBM.load(path, p, gamma, seed=base_seed + t)
    return rbms


# Training sequenziale delle T RBM
def train_all_rbms(X0, T, p, gamma, nv, nh, epochs, batch_size, k, lr, base_seed=0):
    """
    Allena le T RBM (una per step t=1..T) in sequenza, una dopo l'altra.
    """
    rbms, errors = {}, {}
    for t in range(1, T + 1):
        rbm = RBM(nv, nh, t, p, gamma, seed=base_seed + t)
        errors[t] = rbm.train(X0, epochs, batch_size, k, lr, verbose=True)
        rbms[t] = rbm
    return rbms, errors


# Generazione (reverse diffusion, t = T -> 1)
def generate(rbms, gamma, nv, n_samples, start_point=None, start_t=None, k_gen=K_GEN, seed=None):
    """
    rbms: dict {t: RBM} (t=1..T). Usa un rng locale, non quello delle singole RBM.
    """
    rng = np.random.default_rng(seed)
    T_max = max(rbms)
    start_t = start_t or T_max

    if start_point is not None:
        x = np.where(np.tile(start_point, (n_samples, 1)) > 0, 1.0, -1.0).astype(np.float32)
    else:
        x = np.where(rng.random((n_samples, nv)) < 0.5, 1.0, -1.0).astype(np.float32)

    traj = [x]
    x_p = x
    for t in range(start_t, 0, -1):
        field = gamma[t - 1] * x
        # v = np.where(rng.random(x.shape) < 0.5, 1.0, -1.0).astype(np.float32)
        v = x.copy() # collegamento tra layer successivi 
        x, x_p = rbms[t].gibbs(v, field, k_gen)   # dict indicizzato da 1, non da 0
        traj.append(x_p)
    return x_p, traj

def save_generated_grid(samples, out_path, n_cols=8, side=28, S=None):
    """
    Salva una griglia di campioni generati come singolo file PNG.
    """
    n = samples.shape[0]
    n_rows = int(np.ceil(n / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))
    for i, a in enumerate(np.array(ax).ravel()):
        if i < n:
            a.imshow(samples[i].reshape(side, side), cmap="gray")
        a.axis("off")
    if S is not None:
        fig.suptitle(f"Generated samples at step {S}")
    else:
        fig.suptitle(f"Generated samples")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def save_recon_errors_plot(errors, out_path):
    """
    Plotta le curve di recon_error di tutte le RBM sullo stesso grafico e salva su PNG (no show).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in sorted(errors):
        ax.plot(errors[t], label=f"t={t}")
    ax.set_xlabel("Step (batch)")
    ax.set_ylabel("Recon. error (MSE)")
    ax.set_title("Errore di ricostruzione per RBM")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def kgen_from_gamma(gamma, k_max, k_min, sharpness=3.0):
    """
    Determina K_GEN diversi per ogni layer, in funzione del noise schedule scelto.
    """
    g = np.asarray(gamma, dtype=np.float64)
    g_norm = g / (g.max() + 1e-8)          # in [0,1], 0 = field nullo
    weight = (1 - g_norm) ** sharpness      # amplifica il contrasto
    k = k_min + weight * (k_max - k_min)
    return np.maximum(k.astype(int), k_min)   # mai sotto k_min


def generate_last_stage_sweep(rbms,
                              gamma,
                              nv,
                              out_dir,
                              save_generated_grid_fn,
                              n_samples=16,
                              start_point=None,
                              start_t=None,
                              k_first=5,
                              k_last_max=20,
                              seed=None,
                              n_cols=4,
                              side=28):
    """
    Fase 1 (t = start_t..2): cascata standard con warm-start, k_fixed
    passi di Gibbs per ciascuna RBM. Nessun salvataggio intermedio.
 
    Fase 2 (t = 1): singola catena di Gibbs, salva v_p ad ogni passo
    k = 0, 1, ..., k_last_max (k=0 = stato in ingresso, prima di
    qualunque passo di Gibbs sull'ultima RBM).
 
    Ritorna la lista dei path salvati.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    T_max = max(rbms)
    start_t = start_t or T_max
 
    if start_point is not None:
        x = np.where(np.tile(start_point, (n_samples, 1)) > 0, 1.0, -1.0).astype(np.float32)
    else:
        x = np.where(rng.random((n_samples, nv)) < 0.5, 1.0, -1.0).astype(np.float32)
 
    # Fase 1: k_fixed su tutti gli step tranne l'ultimo
 
    for t in range(start_t, 1, -1):
        field = gamma[t - 1] * x
        v = x.copy()                              # warm-start
        x, _ = rbms[t].gibbs(v, field, k_first[t-1])
        print(f"[Timestep {t}, Gibbs step {k_first[t-1]}]")
 
    # Fase 2: catena unica sull'ultimo step (t=1), salvataggio ad ogni k
    rbm1 = rbms[1]
    field = gamma[0] * x
    v = x.copy()                                  # warm-start, stato in ingresso
 
    saved_paths = []
 
    # k = 0: stato prima di qualunque passo di Gibbs sull'ultima RBM
    v_p0 = rbm1.v_prob(rbm1.h_prob(v), field)      # probabilita' "di riferimento" a k=0
    fname0 = os.path.join(out_dir, f"{0:03d}_kgen{0:04d}.png")
    save_generated_grid_fn(v_p0, fname0, n_cols=n_cols, side=side, S=0)
    saved_paths.append(fname0)
    
    for k in range(1, k_last_max + 1):
        h_p = rbm1.h_prob(v)
        h = np.where(rbm1.rng.random(h_p.shape) < h_p, 1.0, -1.0).astype(np.float32)
        v_p = rbm1.v_prob(h, field)
        v = np.where(rbm1.rng.random(v_p.shape) < v_p, 1.0, -1.0).astype(np.float32)
 
        fname = os.path.join(out_dir, f"{k:03d}_kgen{k:04d}.png")
        save_generated_grid_fn(v_p, fname, n_cols=n_cols, side=side, S=k)
        saved_paths.append(fname)
        print(f"[Timestep {1}, Gibbs step {k:2d}] -> saved")
 
    print("Done")
    return saved_paths
    

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training / generazione RBM-diffusion")
    parser.add_argument("--exp_id", type=str, required=True, help="Experiment ID (nome cartella checkpoint)")
    parser.add_argument("--train", action="store_true", default=False, help="Esegui il training")
    parser.add_argument("--generate", action="store_true", default=False, help="Esegui la generazione")
    parser.add_argument("--multigenerate", action="store_true", default=False, help="Esegui la generazione multipla con lo stesso seed")
    args = parser.parse_args()
    
    base_dir = os.path.join("RUNS", args.exp_id)
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    
    if not args.train and not args.generate and not args.multigenerate:
        print("Specifica almeno un argomento: --train o --generate o --multigenerate")
        sys.exit(0)

    p, gamma = noise_schedule(T, kind=NOISE_KIND)

    if args.train:
        if os.path.exists(base_dir):
            print(f"Errore: la directory '{base_dir}' esiste gia'. ")
            sys.exit(1)
        else:
            os.makedirs(base_dir)

        # Salvataggio iperparametri su file txt
        hparam_path = os.path.join(base_dir, "params.txt")
        with open(hparam_path, "w") as f:
            f.write(f"Experiment ID: {args.exp_id}\n")
            f.write("Hyperparameters\n")
            f.write("====================\n")
            f.write(f"T = {T}\n")
            f.write(f"NV = {NV}\n")
            f.write(f"NH = {NH}\n")
            f.write(f"EPOCHS = {EPOCHS}\n")
            f.write(f"BATCH = {BATCH}\n")
            f.write(f"K_TRAIN = {K_TRAIN}\n")
            f.write(f"K_GEN = {K_GEN}\n")
            f.write(f"LR = {LR}\n")
            f.write(f"DATASET = {DATASET}\n")
            f.write(f"NOISE_KIND = {NOISE_KIND}\n")
            f.write(f"SEED = {SEED}\n")

        X_bin, y = load_dataset(DATASET, data_home="DATA/")
        X0 = X_bin.copy()

        rbms, errors = train_all_rbms(
            X0, T, p, gamma, NV, NH, EPOCHS, BATCH, K_TRAIN, LR,
            base_seed=SEED,
        )
        
        save_rbms(rbms, checkpoint_dir)
        errors_plot_path = os.path.join(base_dir, "recon_errors.png")
        save_recon_errors_plot(errors, errors_plot_path)
        
        print("Done")
        print("")
        print(f"RBM checkpoints saved in '{checkpoint_dir}'")
        print(f"Reconstruction error saved in '{errors_plot_path}'")
        print(f"Hyperparameters saved in '{hparam_path}'")

    if args.generate: # deprecated, use multigenerate
        if not args.train:
            if not os.path.exists(checkpoint_dir):
                print(f"Errore: nessun checkpoint trovato in '{checkpoint_dir}' per exp_id='{args.exp_id}'.")
                sys.exit(1)
            rbms = load_rbms(checkpoint_dir, T, p, gamma, base_seed=SEED)

        samples, traj = generate(rbms, gamma, NV, n_samples=16, start_point=None, start_t=T, seed=SEED)

        out_path = os.path.join(base_dir, "generated_samples.png")
        save_generated_grid(samples, out_path)
        print(f"Campioni generati salvati in '{out_path}'")
        
    if args.multigenerate:
        rbms = load_rbms(checkpoint_dir, T, p, gamma, base_seed=SEED)
        # k_schedule = kgen_from_gamma(gamma, k_max=150, k_min=K_TRAIN)
        k_schedule = [10]*T
        generate_last_stage_sweep(
            rbms, gamma, NV,
            out_dir=os.path.join(base_dir, "generated_samples"),
            save_generated_grid_fn=save_generated_grid,
            n_samples=16,
            start_t=T,
            k_first=k_schedule,
            k_last_max=50,
            seed=SEED,
        )