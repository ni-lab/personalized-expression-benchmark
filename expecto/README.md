# Personalized expression benchmark: ExPecto
The ExPecto model is described in the following paper: Jian Zhou, Chandra L. Theesfeld, Kevin Yao, Kathleen M. Chen, Aaron K. Wong,  and Olga G. Troyanskaya, [Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk](https://www.nature.com/articles/s41588-018-0160-6), Nature Genetics (2018).


## Installation
To install the dependencies for the ExPecto model, we recommend using pip. To do so, run the following commands in this directory:
```
python3 -m venv .env_expecto
source .env_expecto/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then, follow the official instructions for installing PyTorch (pytorch <= 1.13) with GPU support from the [official PyTorch site](https://pytorch.org/get-started/previous-versions) for your system.

## Download trained models
To use the ExPecto model, you will need to download pre-trained model weights for the Beluga model that ExPecto relies on. These weights are publicly available from the [official ExPecto repository](https://github.com/FunctionLab/ExPecto/tree/master), and you can obtain them by running the following command in this directory:
```
wget http://deepsea.princeton.edu/media/code/expecto/resources_20190807.tar.gz; tar xf resources_20190807.tar.gz
```

## Predicting expression using reference genome


## Predicting expression using personalized sequences
