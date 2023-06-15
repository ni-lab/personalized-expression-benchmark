# Personalized expression benchmark - Basenji2

## Installation
```
conda create -n basenji_personalized_exp python=3.8
conda activate basenji_personalized_exp
conda env update --file basenji_environment.yml
```

Then, to install tensorflow with GPU support, follow instructions from https://www.tensorflow.org/install/pip for your system.

## Predicting expression using reference genome
To make predictions using the reference genome, we ran the `basenji2_predict_ref.py` script with the following options:
```
basenji2_predict_ref.py ./basenji2/resources/params_human.json ./basenji2/resources/model_human.h5 ./consensus/seqs ./data/gene_list.csv --rc --shifts 1,0,-1 -n 10 --n_uniform -o ./basenji2/out_dir/predict_ref
```

## Predicting expression using personalized sequences
To make predictions using personalized input sequences, we ran the `basenji2_predict_consensus.py` script with the following options:
```
basenji_predict_consensus.py ./basenji2/resources/params_human.json ./basenji2/resources/model_human.h5 ./consensus/seqs ./data/gene_list.csv --rc --shifts 1,0,-1 --all_bins -o ./basenji2/out_dir/predict_consensus
```

Note that the `--all_bins` option is used to predict expression for all bins in the input sequences, which will take up a lot of disk space. If you want to save out expression predictions for only the central bin (which contains thet TSS), you can omit this option.
