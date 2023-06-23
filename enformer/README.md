# Personalized expression benchmark: Enformer
The Enformer model is described in the following paper: Avsec, Ž., Agarwal, V., Visentin, D. et al. [Effective gene expression prediction from sequence by integrating long-range interactions.](https://www.nature.com/articles/s41592-021-01252-x) Nat Methods 18, 1196–1203 (2021). https://doi.org/10.1038/s41592-021-01252-x

## Setup
The Enformer prediction scripts require a GPU and cuda 11.2 or higher.
To create a Conda environment and install dependencies, run the following commands:

```
conda create -n enformer_venv python=3.7 pip
conda activate enformer_venv
pip install -r requirements.txt
```
To install tensorflow with GPU support, follow instructions from https://www.tensorflow.org/install/pip for your system.

## Predicting expression using reference genome
To make predictions using the reference genome, we ran the run_enformer_reference.py script with the following options:
```
python3 enformer/run_enformer_reference.py ./consensus/seqs ./data/gene_list.csv -o out_dir
```

## Predicting expression using personalized sequences
To make predictions using personalized input sequences, we ran the run_enformer_reference.py script with the following options:
```
python3 enformer/run_enformer_all_predictions.py.py ./consensus/seqs ./data/gene_list.csv -o out_dir
```