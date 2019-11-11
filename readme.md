# Code for: Application of Deep Learning for Simulating Time-Series of Oil Futures

This repository contains the code for the master's thesis: Application of Deep Learning for Simulating Time-Series of Oil Futures (link to follow), by Sebastian Ruiz, for the program mathematics, stochastics track, at the University of Amsterdam.

**Note**: The time-series data of oil futures is purposely missing from this project as the data is not publicly available.

## Getting Started

- Set `PATH_ROOT` correctly in `.env` (make a copy of `.env.sample`).

If importing data using the Calibration Library:

1. Add `CalibrationLib` to project interpreter path. In PyCharm: 
Settings -> Project interpreter -> Cog -> Show All -> Select your interpreter -> Click funny folder button -> Add -> Select `CalibrationLib`
2. Set `writableDir` correctly in `data_importer/config.yaml`
3. Run `data_importer/importer_run.py`

Training the models:

1. Set the training and test data in `(root) config.yaml`.
2. To train all the autoencoder models run `autoencoders/autoencoder_run_all.py`.
3. To train all the GAN models run `gans/gan_run_all.py`.

For each model the parameters can be set in the dictionary `ae_params` and `gan_params` for autoencoder parameters and GAN parameters respectively..


## Helper files

- `plotting.py`: Provides function to plot curves. Used to compare autoencoder input and output and show GAN simulations.
- `preprocess_data.py`: Load data from pickle file and does the data pre-processing.
It splits the data into training and test sets, and it applies normalisation, standardisation or log-returns.

## Autoencoders

The code for the models is based on examples found in [keras-autoencoders](https://github.com/snatch59/keras-autoencoders) .

- Standard Autoencoder: Standard autoencoder with encoder and decoder.
- Variational Autoencoder (VAE): The model from [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114).
- Adversarial Autoencoder (AAE): The model from [Adversarial Autoencoders](http://arxiv.org/abs/1511.05644).

## GANs

Popular conditional GAN models. The code is based on examples found in [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN).

- Standard GAN: The generative adversarial network model, from [Generative adversarial networks](http://arxiv.org/abs/1406.2661).
- Wasserstein GAN: The model from [Wasserstein GAN](https://arxiv.org/abs/1701.07875).
- GAN CONV: GAN using convolutional layers.

## LSTMs

The [LSTM Model](http://arxiv.org/abs/1412.3555) is based on example code from [Keras seq-2-seq Signal Prediction](https://github.com/LukeTonin/keras-seq-2-seq-signal-prediction).

## GAIN

The GAIN model, [GAIN: Missing Data Imputation using Generative Adversarial Nets](http://arxiv.org/abs/1806.02920), is based on the Tensorflow example code from [GAIN](https://github.com/jsyoon0823/GAIN).

## Detect Anomalies

The anomaly detection model uses the autoencoder model.

## Classical Models

The classical models include the [Andersen Markov Model](https://ssrn.com/abstract=1138782) and some autoregressive models.
