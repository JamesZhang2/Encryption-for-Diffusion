# Encryption for Diffusion

## BFV Encryption

To run `bfv.py`, we need to install Sage and create a Sage virtual environment:

```bash
brew install --cask sage
sage -python -m venv venv_sage
source venv_sage/bin/activate
sage bfv/bfv.py
```

The BFV scheme is based on this blog post: <https://www.inferati.com/blog/fhe-schemes-bfv>

## Diffusion

To run `cfg-diffusion.ipynb`, install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html). Then install the environment and activate it via
```bash
conda env create -f environment_mac.yml
conda activate diffusion
```
The diffusion model is based on the classifier-free guidance diffusion model from this tutorial: <https://github.com/tsmatz/diffusion-tutorials/blob/master/Readme.md>

## Network
```bash
conda create -n sage python=3.12
conda activate sage
```
If `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.`
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```