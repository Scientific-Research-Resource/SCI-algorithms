# Deep Video Priors for Snapshot Compressive Imaging
Yang Liu, MIT CSAIL, yliu@csail.mit.edu, updated Dec 9, 2019

This repository contains the PyTorch implementation of Deep Video Priors for Snapshot Compressive Imaging, specifically for high-speed (color) videos. The initial Python code for GAP-TV was from [Dr. Xin Yuan](https://www.bell-labs.com/usr/x.yuan) on Aug 7, 2018.

## How to run this code
This code is tested on Ubuntu 18.04 LTS with CUDA 10.0, CuDNN 7.6.2, and PyTorch 1.2.0. It is supposed to work on other platforms (Linux or Windows) with CUDA-enabled GPU(s). 

We use [conda](https://www.anaconda.com/distribution/) to manage the virtual environment and Python packages.

0. Install conda from the Anaconda Distribution https://www.anaconda.com/distribution/ according to the platform.
1. Create the virtual environment with required Python packages via  
`conda env create -f environment.yml`
2. Run a demo test with the `traffic` data via  
`python deep_vprior_sci.py`
3. [Optional] Explore more with the Python Notebook `deep_vprior_sci.ipynb` and the main algorithm code `dvp_linear_inv.py`.

## Acknowledgements
We adopt several image and video denoisers as video priors (under the `./packages/` directory). The `ffdnet` package is from the (unofficial) PyTorch implementation of [FFDNet, TIP'18](https://doi.org/10.1109/TIP.2018.2839891) at https://doi.org/10.5201/ipol.2019.231 (An official PyTorch implementation of FFDNet was out just recently at https://github.com/cszn/KAIR. We believe they should work similarly with respect to the reconstruction process). The `vnlnet` package is from the PyTorch implementation of [VNLnet](https://arxiv.org/abs/1811.12758) at https://github.com/axeldavy/vnlnet. The `fastdvdnet` package is from the PyTorch implementation of [FastDVDnet](https://arxiv.org/abs/1907.01361) at https://github.com/m-tassano/fastdvdnet.