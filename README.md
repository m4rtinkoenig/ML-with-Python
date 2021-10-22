# ML-with-Python [WIP]
In this repository, I create simple projects from classical machine learning, e.g.:
- Classification
    - [x] Logistic Regression
    - [x] k-NN Regression
    - [x] Decision Trees
    - [x] Random Forests
    - [x] Gradient Boosting
    - [ ] Random Projections
- Clustering
    - [ ] Autoencoders
    - [ ] Variational Autoencoders

## Install
Prerequisite: `python >= 3.6`
1. Install `venv`
```bash
venv_name=venv
python -m venv $venv_name  # install venv
source $venv_name/bin/activate  # activate venv
```
2. Install `requirements`
```bash
pip install --upgrade pip  # update pip
pip install -r requirements.txt  # install required packages
```

## Structure
```
.
├── data
│   ├── diabetes
│   └── mnist
├── notebooks
│   ├── Decision Trees.ipynb
│   ├── Gradient Boosting.ipynb
│   ├── KNN Regression.ipynb
│   ├── Logistig Regression.ipynb
│   ├── Random Forest.ipynb
│   └── Random Projections Classifier.ipynb
├── results
│   ├── models
│   └── outputs
├── src
│   ├── __init__.py
│   └── helper
│       ├── __init__.py
│       ├── helper_clean_diabetes.py
│       ├── helper_mnist_download.py
│       └── helper_sklearn_plotting.py
├── README.md
└── requirements.txt
```
Created with tree:
```bash
tree --dirsfirst -I "$(grep -hvE '^$|^#' {~/,,$(git rev-parse --show-toplevel)/}.gitignore|sed 's:/$::'|tr \\n '\|')"
```