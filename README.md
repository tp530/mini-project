# Automating microscope control with Machine Learning

In this repository, I present a CNN-based technique of numerical estimation of defocus of fluorescence microscopy images.

See the Python notebooks below:

- Zernike polynomials and point spread function: [0_main.ipynb](./0_main.ipynb)
- Defocus estimation using Inception v3: [1_inception.ipynb](./1_inception.ipynb)
- Defocus estimation using DenseNet 201: [2_densenet.ipynb](./2_densenet.ipynb)
- Defocus estimation using MobileNet v3 Large: [3_mobilenet.ipynb](./3_mobilenet.ipynb)

Recommended Python version: 3.9
 
Python PIP package pre-requirements: 

```
pip install numpy nptyping scipy matplotlib pillow scikit-learn tensorflow keras
```
