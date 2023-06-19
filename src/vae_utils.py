import random
import pickle
import json
import os

import numpy as np
import scipy.signal as scs
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error

from data.utils import split_ptb

import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, scaler):
        super(VAE, self).__init__()

        self.train_loss = np.array([])
        self.train_mse = np.array([])
        self.train_kl = np.array([])
        self.test_loss = np.array([])
        self.test_mse = np.array([])
        self.test_kl = np.array([])
        self.kl_scaler = np.array([])
        self.scaler = scaler

        self.best_state_dict = None

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def reshape_scaling(self, x, inverse=False, input_length=1000):
        if inverse:
            x = np.asarray(np.split(x, int(x.shape[0] / input_length)))
            x = x.transpose(0, 2, 1)
            return x
        else:
            return np.transpose(x, (0, 2, 1)).reshape(-1, x.shape[1])

    def init_scaler(self, X_train, X_test, input_length):
        X_train = self.reshape_scaling(X_train)
        X_test = self.reshape_scaling(X_test)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        X_train = self.reshape_scaling(X_train, inverse=True, input_length=input_length)
        X_test = self.reshape_scaling(X_test, inverse=True, input_length=input_length)

        return X_train, X_test

    def rescale(self, X, inverse=False, input_length=1000):
        if inverse:
            X = self.reshape_scaling(X)
            X = self.scaler.inverse_transform(X)
            X = self.reshape_scaling(X, inverse=True, input_length=input_length)
        else:
            X = self.reshape_scaling(X)
            X = self.scaler.transform(X)
            X = self.reshape_scaling(X, inverse=True, input_length=input_length)
        return X

    def update_state_dict(self):
        self.best_state_dict = copy.deepcopy(self.state_dict())

    def load_best(self):
        self.load_state_dict(self.best_state_dict)

    def update_loss(self, loss, mse, kl, train=True):
        if train:
            self.train_loss = np.append(self.train_loss, loss)
            self.train_mse = np.append(self.train_mse, mse)
            self.train_kl = np.append(self.train_kl, kl)
        else:
            self.test_loss = np.append(self.test_loss, loss)
            self.test_mse = np.append(self.test_mse, mse)
            self.test_kl = np.append(self.test_kl, kl)

    def update_kl_scaler(self, kl_scaler):
        self.kl_scaler = np.append(self.kl_scaler, kl_scaler)

    def viz_loss(self, path=""):

        top_loss = max(
            np.percentile(self.train_loss, [99])[0],
            np.percentile(self.test_loss, [99])[0],
        )
        min_loss = (
            min(min(self.train_loss), min(self.test_loss))
            - np.percentile(self.test_loss, [50])[0] / 2
        )

        top_mse = max(
            np.percentile(self.train_mse, [99])[0],
            np.percentile(self.test_mse, [99])[0],
        )
        min_mse = (
            min(min(self.train_mse), min(self.test_mse))
            - np.percentile(self.test_mse, [50])[0] / 2
        )

        top_kl = (
            np.percentile(self.train_kl / self.kl_scaler, [50])[0]
            + np.percentile(self.train_kl / self.kl_scaler, [50])[0] / 2
        )
        min_kl = (
            np.percentile(self.train_kl / self.kl_scaler, [50])[0]
            - np.percentile(self.train_kl / self.kl_scaler, [50])[0] / 2
        )

        plt.figure(figsize=(20, 12))
        plt.subplot(2, 1, 1)
        plt.title(f"Loss, Epoch {len(self.train_loss)}")
        plt.plot(self.train_loss, label="train")
        plt.plot(self.test_loss, ":", color="green", label="test")
        plt.plot(np.minimum.accumulate(self.test_loss), color="green", label="test min")
        plt.plot(np.argmin(self.test_loss), min(self.test_loss), "rx")
        plt.ylim(top=top_loss, bottom=min_loss)
        plt.grid()
        plt.legend()
        plt.xlabel("Epochs")
        plt.subplot(2, 2, 3)
        plt.title("mse")
        plt.plot(self.train_mse)
        plt.plot(self.test_mse, ":", color="green", label="test")
        plt.plot(np.minimum.accumulate(self.test_mse), color="green", label="test min")
        plt.plot(np.argmin(self.test_mse), min(self.test_mse), "rx")
        plt.ylim(top=top_mse, bottom=min_mse)
        plt.grid()
        plt.xlabel("Epochs")
        plt.subplot(2, 2, 4)
        plt.title("kl")
        plt.plot(self.train_kl / self.kl_scaler)
        plt.plot(self.test_kl / self.kl_scaler, ":", color="green")
        plt.ylim(top=top_kl, bottom=min_kl)
        plt.grid()
        plt.xlabel("Epochs")
        if path == "":
            plt.show()
        else:
            plt.savefig(path, dpi=150)



class Squeeze_Excite_Block(nn.Module):
    def __init__(self, nb_channel, reduction_ratio=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.reduction_ratio = reduction_ratio
        nb_reduced_channel = nb_channel // reduction_ratio
        self.Dense_1 = nn.Linear(
            in_features=nb_channel, out_features=nb_reduced_channel
        )
        self.Dense_2 = nn.Linear(
            in_features=nb_reduced_channel, out_features=nb_channel
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, nb_channels, _ = input_tensor.size()
        x = input_tensor.mean(dim=-1)
        x = self.relu(self.Dense_1(x))
        x = self.sigmoid(self.Dense_2(x))
        output = torch.mul(input_tensor, x.view(batch_size, nb_channels, 1))
        return output


class CNN_se_VAE(VAE):
    def __init__(
        self,
        scaler,
        input_channels,
        length_ts,
        n_filters_c1,
        size_filters_c1,
        reduction_ratio_c1,
        n_filters_c2,
        size_filters_c2,
        reduction_ratio_c2,
        n_filters_c3,
        size_filters_c3,
        reduction_ratio_c3,
        n_filters_c4,
        size_filters_c4,
        reduction_ratio_c4,
        latent_dim,
    ):
        super(CNN_se_VAE, self).__init__(scaler)

        self.input_channels = input_channels
        self.length_ts = length_ts
        self.n_filters_c1 = n_filters_c1
        self.size_filters_c1 = size_filters_c1
        self.reduction_ratio_c1 = reduction_ratio_c1
        self.n_filters_c2 = n_filters_c2
        self.size_filters_c2 = size_filters_c2
        self.reduction_ratio_c2 = reduction_ratio_c2
        self.n_filters_c3 = n_filters_c3
        self.size_filters_c3 = size_filters_c3
        self.reduction_ratio_c3 = reduction_ratio_c3
        self.n_filters_c4 = n_filters_c4
        self.size_filters_c4 = size_filters_c4
        self.reduction_ratio_c4 = reduction_ratio_c4
        self.latent_dim = latent_dim

        self.bottleneck_length = self.length_ts // 2 ** 3

        # encoder
        self.enc_layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_channels,
                out_channels=self.n_filters_c1,
                kernel_size=self.size_filters_c1,
                padding=self.size_filters_c1 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c1),
            nn.LeakyReLU(),
            Squeeze_Excite_Block(
                nb_channel=self.n_filters_c1, reduction_ratio=self.reduction_ratio_c1
            ),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )
        self.enc_layer2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_filters_c1,
                out_channels=self.n_filters_c2,
                kernel_size=self.size_filters_c2,
                padding=self.size_filters_c2 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c2),
            nn.LeakyReLU(),
            Squeeze_Excite_Block(
                nb_channel=self.n_filters_c2, reduction_ratio=self.reduction_ratio_c2
            ),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )
        self.enc_layer3 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_filters_c2,
                out_channels=self.n_filters_c3,
                kernel_size=self.size_filters_c3,
                padding=self.size_filters_c3 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c3),
            nn.LeakyReLU(),
            Squeeze_Excite_Block(
                nb_channel=self.n_filters_c3, reduction_ratio=self.reduction_ratio_c3
            ),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )
        self.enc_layer4 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_filters_c3,
                out_channels=self.n_filters_c4,
                kernel_size=self.size_filters_c4,
                padding=self.size_filters_c4 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c4),
            nn.LeakyReLU(),
            Squeeze_Excite_Block(
                nb_channel=self.n_filters_c4, reduction_ratio=self.reduction_ratio_c4
            ),
            # nn.Dropout(p=0.2)
        )
        # self.enc_dropout = nn.Dropout(p=0.5)
        self.fc_mu = nn.Linear(
            in_features=self.bottleneck_length * self.n_filters_c4,
            out_features=self.latent_dim,
        )
        self.fc_var = nn.Linear(
            in_features=self.bottleneck_length * self.n_filters_c4,
            out_features=self.latent_dim,
        )

        # decoder
        self.dec_input = nn.Linear(
            in_features=self.latent_dim,
            out_features=self.bottleneck_length * self.n_filters_c4,
        )
        self.dec_layer0 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_filters_c4,
                out_channels=self.n_filters_c3,
                kernel_size=self.size_filters_c3,
                padding=self.size_filters_c3 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c3),
            nn.LeakyReLU(),
            Squeeze_Excite_Block(
                nb_channel=self.n_filters_c3, reduction_ratio=self.reduction_ratio_c3
            ),
            nn.Dropout(p=0.2),
        )
        self.dec_layer1 = nn.Sequential(
            # have stride and output_padding size robust
            nn.ConvTranspose1d(
                in_channels=self.n_filters_c3,
                out_channels=self.n_filters_c2,
                kernel_size=self.size_filters_c2,
                stride=2,
                padding=self.size_filters_c2 // 2,
                output_padding=1,
            ),
            nn.BatchNorm1d(self.n_filters_c2),
            nn.LeakyReLU(),
            Squeeze_Excite_Block(
                nb_channel=self.n_filters_c2, reduction_ratio=self.reduction_ratio_c2
            ),
            nn.Dropout(p=0.2),
        )
        self.dec_layer2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.n_filters_c2,
                out_channels=self.n_filters_c1,
                kernel_size=self.size_filters_c1,
                stride=2,
                padding=self.size_filters_c1 // 2,
                output_padding=1,
            ),
            nn.BatchNorm1d(self.n_filters_c1),
            nn.LeakyReLU(),
            Squeeze_Excite_Block(
                nb_channel=self.n_filters_c1, reduction_ratio=self.reduction_ratio_c1
            ),
            nn.Dropout(p=0.2),
        )
        self.dec_final_layer = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.n_filters_c1,
                out_channels=self.n_filters_c1,
                kernel_size=self.size_filters_c1,
                stride=2,
                padding=self.size_filters_c1 // 2,
                output_padding=1,
            ),
            nn.BatchNorm1d(self.n_filters_c1),
            nn.LeakyReLU(),
            Squeeze_Excite_Block(
                nb_channel=self.n_filters_c1, reduction_ratio=self.reduction_ratio_c1
            ),
            nn.Dropout(p=0.2),
            # add 1x1conv to squeeze tensor depth to 1
            nn.Conv1d(
                in_channels=self.n_filters_c1,
                out_channels=self.input_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        # encode
        mu, log_var = self.encode(x)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        # decode
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

    def decode(self, z):
        x = self.dec_input(z)
        x = x.view(-1, self.n_filters_c4, self.bottleneck_length)
        x = self.dec_layer0(x)
        x = self.dec_layer1(x)
        x = self.dec_layer2(x)
        x = self.dec_final_layer(x)
        return x

    def encode(self, x):
        x = self.enc_layer1(x)
        x = self.enc_layer2(x)
        x = self.enc_layer3(x)
        x = self.enc_layer4(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var


class CNN_VAE(VAE):
    def __init__(
        self,
        scaler,
        input_channels,
        length_ts,
        n_filters_c1,
        size_filters_c1,
        n_filters_c2,
        size_filters_c2,
        n_filters_c3,
        size_filters_c3,
        n_filters_c4,
        size_filters_c4,
        latent_dim,
    ):
        super(CNN_VAE, self).__init__(scaler)

        self.input_channels = input_channels
        self.length_ts = length_ts
        self.n_filters_c1 = n_filters_c1
        self.size_filters_c1 = size_filters_c1
        self.n_filters_c2 = n_filters_c2
        self.size_filters_c2 = size_filters_c2
        self.n_filters_c3 = n_filters_c3
        self.size_filters_c3 = size_filters_c3
        self.n_filters_c4 = n_filters_c4
        self.size_filters_c4 = size_filters_c4
        self.latent_dim = latent_dim

        self.bottleneck_length = self.length_ts // 2 ** 3

        # encoder
        self.enc_layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_channels,
                out_channels=self.n_filters_c1,
                kernel_size=self.size_filters_c1,
                padding=self.size_filters_c1 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )
        self.enc_layer2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_filters_c1,
                out_channels=self.n_filters_c2,
                kernel_size=self.size_filters_c2,
                padding=self.size_filters_c2 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )
        self.enc_layer3 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_filters_c2,
                out_channels=self.n_filters_c3,
                kernel_size=self.size_filters_c3,
                padding=self.size_filters_c3 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c3),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )
        self.enc_layer4 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_filters_c3,
                out_channels=self.n_filters_c4,
                kernel_size=self.size_filters_c4,
                padding=self.size_filters_c4 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c4),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.2)
        )
        # self.enc_dropout = nn.Dropout(p=0.5)
        self.fc_mu = nn.Linear(
            in_features=self.bottleneck_length * self.n_filters_c4,
            out_features=self.latent_dim,
        )
        self.fc_var = nn.Linear(
            in_features=self.bottleneck_length * self.n_filters_c4,
            out_features=self.latent_dim,
        )

        # decoder
        self.dec_input = nn.Linear(
            in_features=self.latent_dim,
            out_features=self.bottleneck_length * self.n_filters_c4,
        )
        self.dec_layer0 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_filters_c4,
                out_channels=self.n_filters_c3,
                kernel_size=self.size_filters_c3,
                padding=self.size_filters_c3 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c3),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        self.dec_layer1 = nn.Sequential(
            # have stride and output_padding size robust
            nn.ConvTranspose1d(
                in_channels=self.n_filters_c3,
                out_channels=self.n_filters_c2,
                kernel_size=self.size_filters_c2,
                stride=2,
                padding=self.size_filters_c2 // 2,
                output_padding=1,
            ),
            nn.BatchNorm1d(self.n_filters_c2),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        self.dec_layer2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.n_filters_c2,
                out_channels=self.n_filters_c1,
                kernel_size=self.size_filters_c1,
                stride=2,
                padding=self.size_filters_c1 // 2,
                output_padding=1,
            ),
            nn.BatchNorm1d(self.n_filters_c1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        self.dec_final_layer = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.n_filters_c1,
                out_channels=self.n_filters_c1,
                kernel_size=self.size_filters_c1,
                stride=2,
                padding=self.size_filters_c1 // 2,
                output_padding=1,
            ),
            nn.BatchNorm1d(self.n_filters_c1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            # add 1x1conv to squeeze tensor depth to 1
            nn.Conv1d(
                in_channels=self.n_filters_c1,
                out_channels=self.input_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        # encode
        mu, log_var = self.encode(x)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        # decode
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

    def decode(self, z):
        x = self.dec_input(z)
        x = x.view(-1, self.n_filters_c4, self.bottleneck_length)
        x = self.dec_layer0(x)
        x = self.dec_layer1(x)
        x = self.dec_layer2(x)
        x = self.dec_final_layer(x)
        return x

    def encode(self, x):
        x = self.enc_layer1(x)
        x = self.enc_layer2(x)
        x = self.enc_layer3(x)
        x = self.enc_layer4(x)
        x = torch.flatten(x, start_dim=1)
        # should probably remove this droupout since it could lead to poor reconstruction
        # x = self.enc_dropout(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var


class CNN_VAE_3(VAE):
    def __init__(
        self,
        scaler,
        input_channels,
        length_ts,
        n_filters_c1,
        size_filters_c1,
        n_filters_c2,
        size_filters_c2,
        n_filters_c3,
        size_filters_c3,
        latent_dim,
    ):
        super(CNN_VAE_3, self).__init__(scaler)

        self.input_channels = input_channels
        self.length_ts = length_ts
        self.n_filters_c1 = n_filters_c1
        self.size_filters_c1 = size_filters_c1
        self.n_filters_c2 = n_filters_c2
        self.size_filters_c2 = size_filters_c2
        self.n_filters_c3 = n_filters_c3
        self.size_filters_c3 = size_filters_c3
        self.latent_dim = latent_dim

        self.bottleneck_length = self.length_ts // 2 ** 2

        # encoder
        self.enc_layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_channels,
                out_channels=self.n_filters_c1,
                kernel_size=self.size_filters_c1,
                padding=self.size_filters_c1 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )
        self.enc_layer2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_filters_c1,
                out_channels=self.n_filters_c2,
                kernel_size=self.size_filters_c2,
                padding=self.size_filters_c2 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )
        self.enc_layer3 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_filters_c2,
                out_channels=self.n_filters_c3,
                kernel_size=self.size_filters_c3,
                padding=self.size_filters_c3 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c3),
            nn.LeakyReLU(),
        )

        self.fc_mu = nn.Linear(
            in_features=self.bottleneck_length * self.n_filters_c3,
            out_features=self.latent_dim,
        )
        self.fc_var = nn.Linear(
            in_features=self.bottleneck_length * self.n_filters_c3,
            out_features=self.latent_dim,
        )

        # decoder
        self.dec_input = nn.Linear(
            in_features=self.latent_dim,
            out_features=self.bottleneck_length * self.n_filters_c3,
        )
        self.dec_layer0 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_filters_c3,
                out_channels=self.n_filters_c2,
                kernel_size=self.size_filters_c2,
                padding=self.size_filters_c2 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c2),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        self.dec_layer1 = nn.Sequential(
            # have stride and output_padding size robust
            nn.ConvTranspose1d(
                in_channels=self.n_filters_c2,
                out_channels=self.n_filters_c1,
                kernel_size=self.size_filters_c1,
                stride=2,
                padding=self.size_filters_c1 // 2,
                output_padding=1,
            ),
            nn.BatchNorm1d(self.n_filters_c1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        self.dec_final_layer = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.n_filters_c1,
                out_channels=self.n_filters_c1,
                kernel_size=self.size_filters_c1,
                stride=2,
                padding=self.size_filters_c1 // 2,
                output_padding=1,
            ),
            nn.BatchNorm1d(self.n_filters_c1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            # add 1x1conv to squeeze tensor depth to 1
            nn.Conv1d(
                in_channels=self.n_filters_c1,
                out_channels=self.input_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        # encode
        mu, log_var = self.encode(x)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        # decode
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

    def decode(self, z):
        x = self.dec_input(z)
        x = x.view(-1, self.n_filters_c3, self.bottleneck_length)
        x = self.dec_layer0(x)
        x = self.dec_layer1(x)
        x = self.dec_final_layer(x)
        return x

    def encode(self, x):
        x = self.enc_layer1(x)
        x = self.enc_layer2(x)
        x = self.enc_layer3(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var


    
class DatasetTrain(Dataset):
    def __init__(self, x, y, transform, output_size):
        assert x.shape[0] == y.shape[0]
        self.x = torch.tensor(x.astype(np.float32))
        self.y = torch.tensor(y.astype(np.float32))
        self.input_size = x.shape[2]
        self.output_size = output_size
        self.nb_variables = x.shape[1]
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            start_index_crop = np.random.randint(0, self.input_size - self.output_size)
            # random crop
            x_ts = self.x[
                index, :, start_index_crop : (start_index_crop + self.output_size)
            ]
        else:
            # center crop
            start_index_crop = self.input_size // 2 - self.output_size // 2
            x_ts = self.x[
                index, :, start_index_crop : (start_index_crop + self.output_size)
            ]
        return x_ts, self.y[index]

    def __len__(self):
        return self.x.shape[0]


class DatasetTrainScaled(Dataset):
    def __init__(self, x, y, transform, scaling_prob, scaling_range, output_size):
        assert x.shape[0] == y.shape[0]
        self.x = torch.tensor(x.astype(np.float32))
        self.y = torch.tensor(y.astype(np.float32))
        self.input_size = x.shape[2]
        self.output_size = output_size
        self.nb_variables = x.shape[1]
        self.transform = transform

        self.scaling_prob = scaling_prob
        self.scaling_inf = scaling_range[0]
        self.scaling_max = scaling_range[1]

    def __getitem__(self, index):
        if self.transform:
            if np.random.rand() < self.scaling_prob:
                x_ts_output = np.empty((self.nb_variables, self.output_size))
                size_scaled = np.random.randint(
                    int(self.output_size * self.scaling_inf),
                    int(self.output_size * self.scaling_max),
                )
                start_index_crop = np.random.randint(0, self.input_size - size_scaled)
                x_ts = (
                    self.x[
                        index, :, start_index_crop : (start_index_crop + size_scaled)
                    ]
                    .detach()
                    .numpy()
                )
                x_ts_output = torch.tensor(scs.resample(x_ts, num=self.output_size, axis=1))
                if x_ts_output.shape[1] != self.output_size:
                    x_ts_output = self.x[
                        index,
                        :,
                        start_index_crop : (start_index_crop + self.output_size),
                    ]
                x_ts = x_ts_output
            else:
                start_index_crop = np.random.randint(
                    0, self.input_size - self.output_size
                )
                # random crop
                x_ts = self.x[
                    index, :, start_index_crop : (start_index_crop + self.output_size)
                ]
        else:
            # center crop
            start_index_crop = self.input_size // 2 - self.output_size // 2
            x_ts = self.x[
                index, :, start_index_crop : (start_index_crop + self.output_size)
            ]

        return x_ts, self.y[index]

    def __len__(self):
        return self.x.shape[0]


# make test dataloader that take 50% overlapping windows over the ts
# return "element-wise maximum" of the predictions whatever this means
class DatasetTest(Dataset):
    def __init__(self, x, y, output_size):
        assert x.shape[0] == y.shape[0]
        self.x = torch.tensor(x.astype(np.float32))
        self.y = torch.tensor(y.astype(np.float32))
        self.input_size = x.shape[2]
        self.output_size = output_size
        self.nb_variables = x.shape[1]

        self.start_indices = np.arange(
            0, self.input_size - self.output_size + 1, self.output_size // 2
        ).tolist()
        if (
            self.input_size
            - (self.start_indices[-1] + self.output_size) / self.output_size
            > 0.33
        ):
            self.start_indices += [self.input_size - self.output_size]
        self.start_indices = np.asarray(self.start_indices)

    def __getitem__(self, index):
        # overlapp crop
        x_ts_all = torch.empty(
            (self.start_indices.shape[0], self.nb_variables, self.output_size)
        )
        for i, start_index in enumerate(self.start_indices):
            x_ts_all[i, :, :] = self.x[
                index, :, start_index : (start_index + self.output_size)
            ]

        return x_ts_all, self.y[index]

    def __len__(self):
        return self.x.shape[0]


def viz_reconstruction_multivariate(model, x, device, path=""):
    ts = x.astype(np.float32)[:, 372:628]
    reconstruction, _, _ = model.forward(
        torch.tensor(ts.reshape((1, 12, ts.shape[1]))).to(device)
    )
    reconstruction = reconstruction.to("cpu").detach().numpy()
    model.eval()
    plt.figure(figsize=(20, 15))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.plot(ts[i, :], label="input")
        plt.plot(reconstruction[0, i, :], label="reconstruct")
        plt.legend()
    if path == "":
        plt.show()
    else:
        plt.savefig(path, dpi=150)
    plt.close("all")


def kl_divergence(mu, logvar):
    """
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    return torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1.0 - logvar, 1))


def fit(model, criterion, data_loader, device, optimizer, kl_scaler):
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_kl = 0.0
    for x, _ in data_loader:
        x = x.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(x)
        mse_loss = criterion(reconstruction, x)
        kl_loss = kl_divergence(mu, logvar) * kl_scaler
        loss = mse_loss + kl_loss
        running_loss += loss.item()
        running_mse += mse_loss.item()
        running_kl += kl_loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss / len(data_loader.dataset)
    train_mse = running_mse / len(data_loader.dataset)
    train_kl = running_kl / len(data_loader.dataset)
    model.update_loss(loss=train_loss, mse=train_mse, kl=train_kl, train=True)
    model.update_kl_scaler(kl_scaler)
    return train_loss, train_mse, train_kl


def validate(model, criterion, data_loader, device, kl_scaler):
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_kl = 0.0
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            reconstruction, mu, logvar = model(x)
            mse_loss = criterion(reconstruction, x)
            kl_loss = kl_divergence(mu, logvar) * kl_scaler
            loss = mse_loss + kl_loss
            running_loss += loss.item()
            running_mse += mse_loss.item()
            running_kl += kl_loss.item()
    val_loss = running_loss / len(data_loader.dataset)
    val_mse = running_mse / len(data_loader.dataset)
    val_kl = running_kl / len(data_loader.dataset)
    model.update_loss(loss=val_loss, mse=val_mse, kl=val_kl, train=False)
    return val_loss, val_mse, val_kl


def get_validation_outputs(model, X_test, Y_test, batch_size, device, model_type):
    # torch dataset generator
    valid_dataset = DatasetTest(
        x=X_test, y=Y_test, output_size=model.__getattribute__("length_ts")
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    x_input = []
    x_reconstruct = []
    x_latent = []
    y_list = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y in valid_dataloader:
            x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3])).to(device)
            
            if model_type in ["contrastive_vae", "vae"]:
                mu, _ = model.encode(x)
            elif model_type in  ["contrastive_ae"]:
                mu = model.encode(x)
            else :
                raise Exception(f"{model_type} not accepted as model_type") 

            reconstruction = model.decode(mu)
            x_input += [x.to("cpu").detach().numpy()]
            x_reconstruct += [reconstruction.to("cpu").detach().numpy()]
            x_latent += [mu.to("cpu").detach().numpy()]
            y_list += [np.repeat(y.to("cpu").detach().numpy(), 7, axis=0)]
    x_input = np.concatenate(x_input, axis=0)
    x_reconstruct = np.concatenate(x_reconstruct, axis=0)
    x_latent = np.concatenate(x_latent, axis=0)
    y_list = np.concatenate(y_list, axis=0)
    return x_input, x_reconstruct, x_latent, y_list


def get_vae_precoler_outputs(model, X, batch_size, device):
    # torch dataset generator
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X.astype(np.float32))
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    x_input = []
    x_reconstruct = []
    x_latent = []
    model.eval()
    with torch.no_grad():
        for x in dataloader:
            x = x[0].to(device)
            _, mu, _ = model(x)
            reconstruction = torch.transpose(model.decode(mu),1,2)
            x_input += [x.to("cpu").detach().numpy()]
            x_reconstruct += [reconstruction.to("cpu").detach().numpy()]
            x_latent += [mu.to("cpu").detach().numpy()]  
    x_input = np.concatenate(x_input, axis=0)
    x_reconstruct = np.concatenate(x_reconstruct, axis=0)
    x_latent = np.concatenate(x_latent, axis=0)
    return x_input, x_reconstruct, x_latent


def get_reconstruction_error(model, X_test, device, model_type):
    X_test_scaled = model.rescale(X_test)
    mse = 0
    model.to(device)
    with torch.no_grad():
        for i_start, i_stop in zip([0,247,503,744],[256,503,759,1000]):
            model.eval()
            ts_input = X_test[:,:,i_start:i_stop]
            ts_input_scaled = X_test_scaled[:,:,i_start:i_stop]
            #try :
            
            if model_type in ["contrastive_vae", "vae"]:
                mu, _ = model.encode(torch.tensor(ts_input_scaled.astype(np.float32)).to(device))
            elif model_type in  ["contrastive_ae"]:
                mu = model.encode(torch.tensor(ts_input_scaled.astype(np.float32)).to(device))
            else :
                raise Exception(f"{model_type} not accepted as model_type") 

            reconstruction = model.decode(mu)
            reconstruction = reconstruction.to("cpu").detach().numpy()
            #except :
            #    mu, _ = model.encode(torch.transpose(torch.tensor(ts_input_scaled.astype(np.float32)),1,2).to(device))
            #    reconstruction = model.decode(mu)
            #    reconstruction = torch.transpose(reconstruction.to("cpu"),1,2).detach().numpy()
                
            reconstruction = model.reshape_scaling(reconstruction, inverse=False, input_length=model.length_ts)
            reconstruction = model.scaler.inverse_transform(reconstruction)
            reconstruction = model.reshape_scaling(reconstruction, inverse=True, input_length=model.length_ts)
            mse += mean_squared_error(y_true=np.reshape(ts_input, (ts_input.shape[0],-1)),y_pred=np.reshape(reconstruction, (reconstruction.shape[0],-1)))
    return mse/4.


def set_seeds(seed, device):
    np.random.seed(seed)  # cpu vars
    torch.manual_seed(seed)  # cpu  vars
    random.seed(seed)  # Python
    if device == "gpu":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def create_dataloaders(training_params, model_params, X, Y, target_y, model):
    # train test split
    X_train, Y_train, X_test, Y_test = split_ptb(
        X, Y, target_y, training_params["test_fold"]
    )
    print(X.shape, Y.shape, target_y.shape)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    # standard scale
    X_train, X_test = model.init_scaler(
        X_train=X_train, X_test=X_test, input_length=1000
    )
    print(X_train.shape, X_test.shape)

    # dataloaders scaled
    train_dataset = DatasetTrainScaled(
        x=X_train,
        y=Y_train,
        transform=True,
        scaling_prob=training_params["scaling_proba"],
        scaling_range=[
            1 - training_params["scaling_param"],
            1 + training_params["scaling_param"],
        ],
        output_size=model_params["length_ts"],
    )
    test_dataset = DatasetTrainScaled(
        x=X_test,
        y=Y_test,
        transform=True,
        scaling_prob=0,
        scaling_range=[0, 0],
        output_size=model_params["length_ts"],
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=training_params["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=training_params["batch_size"], shuffle=False
    )
    return train_dataloader, test_dataloader


def train_vae(
    training_params,
    model_params,
    model,
    output_directory,
    train_dataloader,
    test_dataloader,
    device,
):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    with open(f"{output_directory}/training_params.json", "w") as fp:
        json.dump(training_params, fp, sort_keys=True, indent=4)
    with open(f"{output_directory}/model_params.json", "w") as fp:
        json.dump(model_params, fp, indent=4)

    optimizer = optim.Adam(model.parameters(), lr=training_params["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer, mode="min", patience=training_params["reduce_lr_patience"]
    )
    criterion = nn.MSELoss(reduction="mean")

    best_loss = 1e6
    nb_stag = 0

    for i in range(training_params["max_epochs"]):
        train_loss, train_mse, train_kl = fit(
            model=model,
            criterion=criterion,
            data_loader=train_dataloader,
            device=device,
            optimizer=optimizer,
            kl_scaler=training_params["kl_scaler"],
        )

        test_loss, test_mse, test_kl = validate(
            model=model,
            criterion=criterion,
            data_loader=test_dataloader,
            device=device,
            kl_scaler=training_params["kl_scaler"],
        )
        scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            nb_stag = 0
            model.update_state_dict()
            model.viz_loss(path=f"{output_directory}/loss.png")
            with open(f"{output_directory}/model.pkl", "wb") as output:
                pickle.dump(model, output)
            print(
                f"Epoch {i}, Train loss {train_loss:.6f}, mse {train_mse:.6f}, kl {train_kl} // Test loss {test_loss:.6f}, mse {test_mse:.6f}, kl : {test_kl}, best epoch"
            )
        else:
            nb_stag += 1
            print(
                f"Epoch {i}, Train loss {train_loss:.6f}, mse {train_mse:.6f}, kl {train_kl} // Test loss {test_loss:.6f}, mse {test_mse:.6f}, kl : {test_kl}"
            )
        if nb_stag == training_params["early_stopping"]:
            break

    with open(f"{output_directory}/model.pkl", "wb") as output:
        pickle.dump(model, output)

    return model
