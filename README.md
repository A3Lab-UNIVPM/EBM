## Energy Based Models: Restricted Boltzmann Machines

Two NumPy implementations of Bernoulli-Bernoulli RBMs on MNIST / Fashion-MNIST, exploring CD-k training and Gibbs-sampling generation dynamics.

### `monolithic_RBM.ipynb`: Single RBM, sampling dynamics analysis

A single RBM (784 visible / 256 hidden units) trained with CD-k, followed by an empirical study of Gibbs-chain generation under different initialization strategies.

The notebook tracks free energy along each chain to characterize mode basins (depth/width) and documents observed **mode collapse** (e.g., convergence toward the "0" digit or the Coat/Pullover/Shirt cluster on Fashion-MNIST), discussing it in relation to the known bias of short-k CD training.

### `denoising_RBM.py`: Cascaded RBMs as a discrete diffusion model

A cascade of independent RBMs, one per timestep `t`, implementing a discrete (binary/Bernoulli) forward diffusion process.

Each RBM is trained with CD-k to reverse its corresponding forward corruption step, conditioned via an external field term derived from the noise schedule (`linear`, `quadratic`, `sigmoid`, `cosine`, `geometric).

Includes checkpointing, reconstruction-error tracking per RBM, and a last-stage Gibbs sweep for inspecting the final denoising step in detail.
