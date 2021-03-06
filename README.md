# Analysing recurrent neural network training

Companion code to the 2020 AISTATS paper: 
''Beyond exploding and vanishing gradients: analysing RNN training using attractors and smoothness"
https://arxiv.org/pdf/1906.08482.pdf.

```
@inproceedings{ribeiro_beyond_2020,
author = {Ribeiro, Ant\^{o}nio H. and Tiels, Koen and Aguirre,
Luis A. and Sch\"on, Thomas B. },
title = {Beyond exploding and vanishing gradients: analysing
RNN training using attractors and smoothness},
year = {2020},
booktitle = {Proceedings of the 23rd International Conference
on Artificial Intelligence and Statistics (AISTATS).},
publisher = {PMLR},
volume = {108}
}
```


## Download and Usage


To load this repository run:
```bash
git clone git@github.com:antonior92/attractors-and-smoothness-RNN.git
# or, alternatively: git clone https://github.com/antonior92/attractors-and-smoothness-RNN.git
```

### Submodules

This package have [expRNN](https://github.com/Lezcano/expRNN) as a submodule. To
load this submodule run (from inside the repository):
```bash
git submodule init
git submodule update
```
or, alternatively, add the option ``--recurse-submodules`` in git clone.

### Dependencies

This package was tested in Python 3.6 and for PyTorch 1.1. Other dependencies include 
numpy, scipy, matplotlib and tqdm. Check ``requirements.txt``.

### Required datasets

We use the "wikitext-2" dataset in the word-level language model example (a full description of the dataset is available
[here](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/)).

The dataset is small and we save it inside this git repository (`word-language-model/wikitext-2/`) using 
git-lfs extension (available [here](https://git-lfs.github.com/)).


## Repository structure

This repository contain a folder for each of the experiments described in the paper. 
Each folder contain scripts for training and analysing RNNs in the tasks of:
- ``sine-generator``:  modeling sine-waves with varying frequencies;
- ``word-language-model``: obtaining a word-level language model from wikitext-2 dataset;
- ``sequence-classification``: classification of sequences according to a few relevant symbols.


## Model training

To train the model run:
```bash
cd sine-generator  # or "word-language-model" or "sequence-classification"
python train.py
```
there is the option ``--cuda`` which should enable the use of the GPU. The script will
should create a folder ``output-YYYY-MM-DD HH:MM:SS.MMMMMM`` (with the date and time the script
was called) inside the current directory and save information inside it. Alternatively,
it is possible to pass an option ``--folder [FOLDERNAME]`` which specifies the name of the folder.
To see all options check options run:
```bash
python train.py --help
```
which should print all possible command line options. Which include different architecture configuration
and training specifications. One important option is ``--interval_orbit N``, which save points
every ``N``-th epoch, which latter can be used for generating the biffurcation diagrams (a.k.a. orbit diagram)
such as the one bellow: (as we will describe ahead)
![img/biffurcation_diagrams.png](img/biffurcation_diagrams.png)


#### Training output
The folder created during training will have the structure:
```
FOLDERNAME/
    history.csv
    model.pth
    train.pyconfig.json
    (orbit_per_epoch_diagram.pth)
```
where ``history.csv`` is a column separated text file containing: training and validation loss; learning rate and
other metrics, for the model during the training procedure (per epoch). ``model.pth`` is a binary file containing
the weights of the model that was obtained. ``train.pyconfig.json`` is a json file containing the configuration
(i.e., the options passed from the command line) that were used to obtain such model. Finally,
``orbit_per_epoch_diagram.pth`` is a file that will be create only if the option ``--interval_orbit N``
is passed during train. This binary file contains points saved during training every ``N``-th epoch, these points
can be used for generating the biffurcation diagram (a.k.a. orbit diagram).

#### Pretrained models and datapoint

Pretrained models and datapoints can be downloaded from: 

- Zenodo: https://doi.org/10.5281/zenodo.3834894
- Dropbox: https://www.dropbox.com/s/bgx68aoehup1lh0/beyond-explod-and-vanish-grad.zip?dl=0

The above dropbox link can also be used to download the files from the command line, 
```
wget https://www.dropbox.com/s/bgx68aoehup1lh0/beyond-explod-and-vanish-grad.zip?dl=0
```

## Generating and plotting bifurcation diagrams

As mentioned before, during training we have the option to save intermediary 
datapoints in the pytorch binary file ``orbit_per_epoch_diagram.pth``. This file
contains a 5D-tensor `trajectory` with the dimensions 
`(seq_len, n_seq, n_i, n_o, n_ep)`
and can be used to generate a bifurcation diagram (aka orbit diagram), which depicts
the steady-state behavior of the system along the epochs. So this tensor contain
the model trajectory for a constant output (possibly, after a burnout period) for:
 - `n_i` different constant inputs applied to the system;
 - for `n_o` different (transformed) outputs measure points; 
 - for `n_seq` different initializations
 - for a total of `n_ep`;
 - and for a total trajectory length of `seq_len`.

#### Plotting the bifurcation diagram

We provide a lightweight script `plot_orbit_diagram.py` to generate matplotlib plots of this bifurcation diagram 
from the saved file. This script might be called from the command line as:
```bash
python plot_orbit_diagram.py $PATH
```
where path is the path to the `.pth` file containing the datapoints. The option `--help` should give a list of the 
parameters that can be passed to the plot.


We give a few examples bellow using the pretrained models and the datapoints we provide 
([link](https://doi.org/10.5281/zenodo.3834894);
 [link](https://www.dropbox.com/s/bgx68aoehup1lh0/beyond-explod-and-vanish-grad.zip?dl=0)). 
Let ``DATAPOINT_FOLDER`` point to the location of the top folder after the zip extraction. For instance:
```bash
unzip beyond-explod-and-vanish-grad.zip
DATAPOINTS=./beyond-explod-and-vanish-grad
```


Setting the command `-t orbit2d` one can generate 2 dimensional bifurcation plots, for each first dimension correspond to
the state and the other to the first difference. Using this option it is possible to generate figures similar to 
Figure 4(a) in the paper. For instance:
```bash
python plot_orbit_diagram.py $DATAPOINTS/sine-generator/lstm/orbit_per_epoch_diagram.pth -t orbit2d -b 300 -ie 30 -fe 300
```
Expected output:
![lstm-sine](./img/LSTM_sine_classification.png)

It is also possible to generate one-dimensional bifurcation plots by using the option `-t orbit1d`. For instance,
we can generate something similar to Figure 8 (c) in the paper, by using:
```bash
python plot_orbit_diagram.py $DATAPOINTS/word-language-model/oRNN/orbit_per_epoch_diagram.pth -t orbit1d -b 300
```
Expected output:
![ornn-wlm](./img/oRNN_world_language_model.png)
