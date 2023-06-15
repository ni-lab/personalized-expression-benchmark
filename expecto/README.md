# Personalized expression benchmark: ExPecto
The ExPecto model is described in the following paper: Jian Zhou, Chandra L. Theesfeld, Kevin Yao, Kathleen M. Chen, Aaron K. Wong,  and Olga G. Troyanskaya, [Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk](https://www.nature.com/articles/s41588-018-0160-6), Nature Genetics (2018).

## Installation
To install the dependencies for the ExPecto model, we recommend using conda. To do so, run the following commands in this directory:
```
conda create -n expecto_personalized_exp python=3.8
conda activate expecto_personalized_exp
conda env update --file expecto_environment.yml
```

Then, follow the official instructions for installing PyTorch (pytorch <= 1.13) with GPU support from the [official PyTorch site](https://pytorch.org/get-started/previous-versions) for your system.

## Download trained models
To use the ExPecto model, you will need to download pre-trained model weights for the Beluga model that ExPecto relies on. These weights are publicly available from the [official ExPecto repository](https://github.com/FunctionLab/ExPecto/tree/master), and you can obtain them by running the following command in this directory:
```
wget http://deepsea.princeton.edu/media/code/expecto/resources_20190807.tar.gz; tar xf resources_20190807.tar.gz
```

## Predicting expression using reference genome
To make predictions using the reference genome, we ran the `expecto_predict_ref.py` script with the following options:
```
python3 expecto/expecto_predict_ref.py ./expecto/models/allhistones2000.1.fixed.all.pseudocount0.0001.lambda100.round100.basescore2.Cells_EBV-transformed_lymphocytes.save ./consensus/seqs ./data/gene_list.csv --beluga_model ./expecto/resources/deepsea.beluga.pth --batch_size 1024 -o ./expecto/out_dir/expecto_ref_preds
```

## Predicting expression using personalized sequences
To make predictions using personalized input sequences, we ran the `expecto_predict_consensus.py` script with the following options:
```
python3 expecto/expecto_predict_consensus.py ./expecto/models/allhistones2000.1.fixed.all.pseudocount0.0001.lambda100.round100.basescore2.Cells_EBV-transformed_lymphocytes.save ./consensus/seqs ./data/gene_list.csv --beluga_model ./expecto/resources/deepsea.beluga.pth --batch_size 1024 -o ./expecto/out_dir/expecto_consensus_preds
```
However, due to high computational requirements, we distributed this script across 200 CPU nodes, each of which predicted expression for a subset of the genes in the gene list. To do this, we used the `--num_chunks` option to split the gene list into 200 chunks, and then ran the following command for each chunk:
```
python3 expecto/expecto_predict_consensus.py ./expecto/models/allhistones2000.1.fixed.all.pseudocount0.0001.lambda100.round100.basescore2.Cells_EBV-transformed_lymphocytes.save ./consensus/seqs ./data/gene_list.csv --beluga_model ./expecto/resources/deepsea.beluga.pth --batch_size 1024 -o ./expecto/out_dir/expecto_consensus_preds --num_chunks 200 --chunk_index ${i}
```
where `i` denotes the index of the chunk to run for that job. The resulting predictions take up ~8.1TB of space, so we recommend running this script on a cluster with a large amount of storage space.
