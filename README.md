# personalized-expression-benchmark
Official repository for the paper "Personal transcriptome variation is poorly explained by current genomic deep learning models" <-- insert link later

## Installation
We evaluated gene expression prediction performance using 4 models: Enformer, Basenji2, ExPecto, and Xpresso. We installed each model in a separate conda environment. To find instructions for installation, see the corresponding README.md file in each model's directory.

## Data
To make predictions on the reference genome and personalized consensus sequences, you will need to download the reference and consensus sequences for each gene and place these sequences in the `consensus/seqs` directory. We provide the following script for doing so:
```
# TODO: make the script for this
bash download_consensus_seqs.sh
```
