## Committing Jupyter notebooks

To commit notebooks, first execute the following to strip all notebooks of output.

python3 -m nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb **/*.ipynb

## Equation numbering in Juypter

# install
pip install jupyter_contrib_nbextensions

# enable
jupyter contrib nbextension install --user
jupyter nbextension enable equation-numbering/main

# activate
MathJax.Hub.Config({
    TeX: { equationNumbers: { autoNumber: "AMS" } }
});
