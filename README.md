# Encryption for Diffusion

## BFV Encryption

- To run `bfv.py`, we need to install Sage and create a Sage virtual environment:

```bash
brew install --cask sage
sage -python -m venv venv_sage
source venv_sage/bin/activate
sage bfv/bfv.py
```

- The BFV scheme is based on this blog post: <https://www.inferati.com/blog/fhe-schemes-bfv>
