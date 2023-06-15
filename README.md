# personalized-expression-benchmark
Official repository for the manuscript
"Personal transcriptome variation is poorly explained by current genomic deep learning models"

TODO: insert link later

## Installation
We evaluated gene expression prediction performance using 4 models: Enformer, Basenji2, ExPecto, and Xpresso. We installed each model in a separate python environment. To find instructions for installing dependencies for each model, see the corresponding README.md file in each model's directory.

## Data
To make predictions on the reference genome and personalized consensus sequences, you will need to generate the reference and consensus sequences for each gene and place these sequences in the `consensus/seqs` directory. We provide the following script for doing so:
```
# TODO: make the script for this
```

## Evaluating gene expression prediction performance
Within each directory, we provide instructions for evaluating gene expression prediction performance using the corresponding model. See the README.md file in each directory for more details.
