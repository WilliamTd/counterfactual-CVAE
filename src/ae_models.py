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

from src.vae_utils import kl_divergence, VAE


class BaseAE(nn.Module):
    def __init__(self, scaler):
        super(BaseAE, self).__init__()

        self.train_loss = np.array([])
        self.train_mse = np.array([])
        self.test_loss = np.array([])
        self.test_mse = np.array([])
        self.scaler = scaler

        self.best_state_dict = None

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

    def update_loss(self, loss, mse, train=True):
        if train:
            self.train_loss = np.append(self.train_loss, loss)
            self.train_mse = np.append(self.train_mse, mse)
        else:
            self.test_loss = np.append(self.test_loss, loss)
            self.test_mse = np.append(self.test_mse, mse)


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
        plt.subplot(2, 1, 2)
        plt.title("mse")
        plt.plot(self.train_mse)
        plt.plot(self.test_mse, ":", color="green", label="test")
        plt.plot(np.minimum.accumulate(self.test_mse), color="green", label="test min")
        plt.plot(np.argmin(self.test_mse), min(self.test_mse), "rx")
        plt.ylim(top=top_mse, bottom=min_mse)
        plt.grid()
        plt.xlabel("Epochs")
        if path == "":
            plt.show()
        else:
            plt.savefig(path, dpi=150)


class CNN_AE(BaseAE):
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
        super(CNN_AE, self).__init__(scaler)

        self.input_channels = input_channels
        self.length_ts = length_ts
        self.n_filters_c1 = n_filters_c1
        self.size_filters_c1 = size_filters_c1
        self.n_filters_c2 = n_filters_c2
        self.size_filters_c2 = size_filters_c2
        self.n_filters_c3 = n_filters_c3
        self.size_filters_c3 = size_filters_c3
        self.latent_dim = latent_dim
        
        self.train_contrastive = np.array([])
        self.test_contrastive = np.array([])
        
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

        self.code = nn.Linear(
            in_features=self.bottleneck_length * self.n_filters_c3,
            out_features=self.latent_dim,
        )
        # decoder
        self.dec_input = nn.Linear(
            in_features=self.latent_dim,
            out_features=self.bottleneck_length * self.n_filters_c3,
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
        code = self.encode(x)
        # decode
        reconstruction = self.decode(code)
        return reconstruction, code

    def decode(self, z):
        x = self.dec_input(z)
        x = x.view(-1, self.n_filters_c3, self.bottleneck_length)
        x = self.dec_layer1(x)
        x = self.dec_layer2(x)
        x = self.dec_final_layer(x)
        return x

    def encode(self, x):
        x = self.enc_layer1(x)
        x = self.enc_layer2(x)
        x = self.enc_layer3(x)
        x = torch.flatten(x, start_dim=1)
        code = self.code(x)
        return code
    
    def update_loss(self, loss, mse, contrastive, train=True):
        if train:
            self.train_loss = np.append(self.train_loss, loss)
            self.train_mse = np.append(self.train_mse, mse)
            self.train_contrastive = np.append(self.train_contrastive, contrastive)
        else:
            self.test_loss = np.append(self.test_loss, loss)
            self.test_mse = np.append(self.test_mse, mse)
            self.test_contrastive = np.append(self.test_contrastive, contrastive)
            
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

        plt.figure(figsize=(20, 12))
        plt.subplot(2, 2, 1)
        plt.title(f"Loss, Epoch {len(self.train_loss)}")
        plt.plot(self.train_loss, label="train")
        plt.plot(self.test_loss, ":", color="green", label="test")
        plt.plot(np.minimum.accumulate(self.test_loss), color="green", label="test min")
        plt.plot(np.argmin(self.test_loss), min(self.test_loss), "rx")
        plt.ylim(top=top_loss, bottom=min_loss)
        plt.grid()
        plt.legend()
        plt.xlabel("Epochs")
        plt.subplot(2, 2, 2)
        plt.title("mse")
        plt.plot(self.train_mse)
        plt.plot(self.test_mse, ":", color="green", label="test")
        plt.plot(np.minimum.accumulate(self.test_mse), color="green", label="test min")
        plt.plot(np.argmin(self.test_mse), min(self.test_mse), "rx")
        plt.ylim(top=top_mse, bottom=min_mse)
        plt.grid()
        plt.xlabel("Epochs")
        plt.subplot(1, 2, 2)
        plt.title("contrastive")
        plt.plot(self.train_contrastive)
        plt.plot(self.test_contrastive, ":", color="green", label="test")
        plt.plot(
            np.minimum.accumulate(self.test_contrastive),
            color="green",
            label="test min",
        )
        plt.plot(np.argmin(self.test_contrastive), min(self.test_contrastive), "rx")
        plt.grid()
        plt.xlabel("Epochs")

        if path == "":
            plt.show()
        else:
            plt.savefig(path, dpi=150)
    
    def viz_reconstruction_multivariate(self, x, device, path=""):
        ts = x.astype(np.float32)[:, 372:628]
        self.eval()
        with torch.no_grad():
            reconstruction, _ = self.forward(
                torch.tensor(ts.reshape((1, 12, ts.shape[1]))).to(device)
            )
        reconstruction = reconstruction.to("cpu").detach().numpy()

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
    
    

def fit_ae(model, criterion, data_loader, device, optimizer):
    model.train()
    running_mse = 0.0
    for x, _ in data_loader:
        x = x.to(device)
        optimizer.zero_grad()
        reconstruction, _ = model(x)
        mse_loss = criterion(reconstruction, x)
        running_mse += mse_loss.item()
        mse_loss.backward()
        optimizer.step()
    train_mse = running_mse / len(data_loader.dataset)
    model.update_loss(loss=train_mse, mse=train_mse, kl=0, train=True)
    model.update_kl_scaler(1)
    return train_mse


def validate_ae(model, criterion, data_loader, device):
    model.eval()
    running_mse = 0.0
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            reconstruction, _ = model(x)
            mse_loss = criterion(reconstruction, x)
            running_mse += mse_loss.item()
    val_mse = running_mse / len(data_loader.dataset)
    model.update_loss(loss=val_mse, mse=val_mse, kl=1, train=False)
    return val_mse

def fit_salient_contrastive_ae(
    model,
    criterion,
    sup_con_loss,
    data_loader,
    device,
    optimizer,
    contrastive_scaler,
    nb_salient_features,
):
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_contrastive = 0.0
    for x, y in data_loader:
        y = y.argmax(axis=1).to(device)
        x = torch.cat([x[0].to(device), x[1].to(device)], dim=0)
        nb_x = y.shape[0]
        
        optimizer.zero_grad()
        reconstruction, code= model(x)
        mse_loss = criterion(reconstruction, x)

        f1, f2 = torch.split(torch.nn.functional.normalize(code[:,-nb_salient_features:]), [nb_x, nb_x], dim=0)

        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        contrastive_loss = sup_con_loss(features, y) * contrastive_scaler
        loss = mse_loss + contrastive_loss
        running_loss += loss.item()
        running_mse += mse_loss.item()
        running_contrastive += contrastive_loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = running_loss / len(data_loader.dataset)
    train_mse = running_mse / len(data_loader.dataset)
    train_contrastive = running_contrastive / len(data_loader.dataset)
    model.update_loss(
        loss=train_loss,
        mse=train_mse,
        contrastive=train_contrastive,
        train=True,
    )
    return train_loss, train_mse, train_contrastive


def validate_salient_contrastive_ae(
    model, criterion, sup_con_loss, data_loader, device, contrastive_scaler, nb_salient_features
):
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_contrastive = 0.0
    with torch.no_grad():
        for x, y in data_loader:
            y = y.argmax(axis=1).to(device)
            x = torch.cat([x[0].to(device), x[1].to(device)], dim=0)
            nb_x = y.shape[0]
            reconstruction, code = model(x)
            mse_loss = criterion(reconstruction, x)
            f1, f2 = torch.split(torch.nn.functional.normalize(code[:,-nb_salient_features:]), [nb_x, nb_x], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            contrastive_loss = sup_con_loss(features, y) * contrastive_scaler
            loss = mse_loss + contrastive_loss
            running_loss += loss.item()
            running_mse += mse_loss.item()
            running_contrastive += contrastive_loss.item()

    val_loss = running_loss / len(data_loader.dataset)
    val_mse = running_mse / len(data_loader.dataset)
    val_contrastive = running_contrastive / len(data_loader.dataset)
    model.update_loss(
        loss=val_loss,
        mse=val_mse,
        contrastive=val_contrastive,
        train=False,
    )
    return val_loss, val_mse, val_contrastive