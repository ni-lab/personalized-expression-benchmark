# Personalized expression benchmark: Basenji2
The Basenji2 model is described in the following paper: Kelley, D.R., [Cross-species regulatory sequence activity prediction](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008050), PLoS computational biology (2020).


## Installation
To install the dependencies for the Basenji2 model, we recommend using conda. To do so, run the following commands in this directory:
```
conda create -n basenji_personalized_exp python=3.8
conda activate basenji_personalized_exp
conda env update --file basenji_environment.yml
```

Then, to install tensorflow with GPU support, follow instructions from https://www.tensorflow.org/install/pip for your system.

## Download trained models
To use the Basenji2 model, you will need to download the publicly available pre-trained model weights from the [official Basenji repository](https://github.com/calico/basenji/tree/master). You can do this by running the following command in this directory:
```
wget https://storage.googleapis.com/basenji_barnyard2/model_human.h5 -O resources/model_human.h5
```
which places the model weights in the `resources` directory.

## Predicting expression using reference genome
To make predictions using the reference genome, we ran the `basenji2_predict_ref.py` script with the following options:
```
basenji2_predict_ref.py ./basenji2/resources/params_human.json ./basenji2/resources/model_human.h5 ./consensus/seqs ./data/gene_list.csv --rc --shifts 1,0,-1 -n 10 --n_uniform -o ./basenji2/out_dir/predict_ref
```

## Predicting expression using personalized sequences
To make predictions using personalized input sequences, we ran the `basenji2_predict_consensus.py` script with the following options:
```
basenji_predict_consensus.py ./basenji2/resources/params_human.json ./basenji2/resources/model_human.h5 ./consensus/seqs ./data/gene_list.csv --rc --shifts 1,0,-1 -n 10 --n_uniform --all_bins -o ./basenji2/out_dir/predict_consensus
```

Note that the `--all_bins` option is used to predict expression for all bins in the input sequences, which will take up a lot of disk space. If you want to save out expression predictions for only the central bin (which contains thet TSS), you can omit this option.
