import numpy as np
import scipy.signal as scs
import pandas as pd
import os
import sys
import pickle
import json
import random
from time import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
#import umap.umap_ as umap

from src.vae_utils import kl_divergence, VAE



class SupConLoss(nn.Module):
    """
    Author : https://github.com/HobbitLong/SupContrast
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
    
    
    
def fit_contrastive(
    model,
    criterion,
    sup_con_loss,
    data_loader,
    device,
    optimizer,
    kl_scaler,
    contrastive_scaler,
):
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_kl = 0.0
    running_contrastive = 0.0
    for x, y in data_loader:
        y = y.argmax(axis=1).to(device)
        x = torch.cat([x[0].to(device), x[1].to(device)], dim=0)
        nb_x = y.shape[0]
        optimizer.zero_grad()
        reconstruction, z, mu, log_var = model(x)
        mse_loss = criterion(reconstruction, x)
        kl_loss = kl_divergence(mu, log_var) * kl_scaler

        f1, f2 = torch.split(torch.nn.functional.normalize(z), [nb_x, nb_x], dim=0)

        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        contrastive_loss = sup_con_loss(features, y) * contrastive_scaler
        loss = mse_loss + kl_loss + contrastive_loss
        running_loss += loss.item()
        running_mse += mse_loss.item()
        running_kl += kl_loss.item()
        running_contrastive += contrastive_loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = running_loss / len(data_loader.dataset)
    train_mse = running_mse / len(data_loader.dataset)
    train_kl = running_kl / len(data_loader.dataset)
    train_contrastive = running_contrastive / len(data_loader.dataset)
    model.update_loss(
        loss=train_loss,
        mse=train_mse,
        kl=train_kl,
        contrastive=train_contrastive,
        train=True,
    )
    model.update_kl_scaler(kl_scaler)
    return train_loss, train_mse, train_kl, train_contrastive


def validate_contrastive(
    model, criterion, sup_con_loss, data_loader, device, kl_scaler, contrastive_scaler
):
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_kl = 0.0
    running_contrastive = 0.0
    with torch.no_grad():
        for x, y in data_loader:
            y = y.argmax(axis=1).to(device)
            x = torch.cat([x[0].to(device), x[1].to(device)], dim=0)
            nb_x = y.shape[0]
            reconstruction, z, mu, log_var = model(x)
            mse_loss = criterion(reconstruction, x)
            kl_loss = kl_divergence(mu, log_var) * kl_scaler
            f1, f2 = torch.split(torch.nn.functional.normalize(z), [nb_x, nb_x], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            contrastive_loss = sup_con_loss(features, y) * contrastive_scaler
            loss = mse_loss + kl_loss + contrastive_loss
            running_loss += loss.item()
            running_mse += mse_loss.item()
            running_kl += kl_loss.item()
            running_contrastive += contrastive_loss.item()

    val_loss = running_loss / len(data_loader.dataset)
    val_mse = running_mse / len(data_loader.dataset)
    val_kl = running_kl / len(data_loader.dataset)
    val_contrastive = running_contrastive / len(data_loader.dataset)
    model.update_loss(
        loss=val_loss, mse=val_mse, kl=val_kl, contrastive=val_contrastive, train=False
    )
    return val_loss, val_mse, val_kl, val_contrastive

def fit_salient_contrastive(
    model,
    criterion,
    sup_con_loss,
    data_loader,
    device,
    optimizer,
    kl_scaler,
    contrastive_scaler,
    nb_salient_features,
):
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_kl = 0.0
    running_contrastive = 0.0
    for x, y in data_loader:
        y = y.argmax(axis=1).to(device)
        x = torch.cat([x[0].to(device), x[1].to(device)], dim=0)
        nb_x = y.shape[0]
        
        optimizer.zero_grad()
        reconstruction, z, mu, log_var = model(x)
        mse_loss = criterion(reconstruction, x)
        kl_loss = kl_divergence(mu, log_var) * kl_scaler

        f1, f2 = torch.split(torch.nn.functional.normalize(z[:,-nb_salient_features:]), [nb_x, nb_x], dim=0)

        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        contrastive_loss = sup_con_loss(features, y) * contrastive_scaler
        loss = mse_loss + kl_loss + contrastive_loss
        running_loss += loss.item()
        running_mse += mse_loss.item()
        running_kl += kl_loss.item()
        running_contrastive += contrastive_loss.item()
        loss.backward()
        optimizer.step()
        # print('contrastive_loss', contrastive_loss)
        # print('running_contrastive', running_contrastive)
    train_loss = running_loss / len(data_loader.dataset)
    train_mse = running_mse / len(data_loader.dataset)
    train_kl = running_kl / len(data_loader.dataset)
    train_contrastive = running_contrastive / len(data_loader.dataset)
    model.update_loss(
        loss=train_loss,
        mse=train_mse,
        kl=train_kl,
        contrastive=train_contrastive,
        train=True,
    )
    model.update_kl_scaler(kl_scaler)
    return train_loss, train_mse, train_kl, train_contrastive


def validate_salient_contrastive(
    model, criterion, sup_con_loss, data_loader, device, kl_scaler, contrastive_scaler, nb_salient_features
):
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_kl = 0.0
    running_contrastive = 0.0
    with torch.no_grad():
        for x, y in data_loader:
            y = y.argmax(axis=1).to(device)
            x = torch.cat([x[0].to(device), x[1].to(device)], dim=0)
            nb_x = y.shape[0]
            reconstruction, z, mu, log_var = model(x)
            mse_loss = criterion(reconstruction, x)
            kl_loss = kl_divergence(mu, log_var) * kl_scaler
            f1, f2 = torch.split(torch.nn.functional.normalize(z[:,-nb_salient_features:]), [nb_x, nb_x], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            contrastive_loss = sup_con_loss(features, y) * contrastive_scaler
            loss = mse_loss + kl_loss + contrastive_loss
            running_loss += loss.item()
            running_mse += mse_loss.item()
            running_kl += kl_loss.item()
            running_contrastive += contrastive_loss.item()

    val_loss = running_loss / len(data_loader.dataset)
    val_mse = running_mse / len(data_loader.dataset)
    val_kl = running_kl / len(data_loader.dataset)
    val_contrastive = running_contrastive / len(data_loader.dataset)
    model.update_loss(
        loss=val_loss, mse=val_mse, kl=val_kl, contrastive=val_contrastive, train=False
    )
    return val_loss, val_mse, val_kl, val_contrastive




def fit_salient_contrastive_nosalient_kl(
    model,
    criterion,
    sup_con_loss,
    data_loader,
    device,
    optimizer,
    kl_scaler,
    contrastive_scaler,
    nb_salient_features,
):
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_kl = 0.0
    running_contrastive = 0.0
    for x, y in data_loader:
        y = y.argmax(axis=1).to(device)
        x = torch.cat([x[0].to(device), x[1].to(device)], dim=0)
        nb_x = y.shape[0]
        
        optimizer.zero_grad()
        reconstruction, z, mu, log_var = model(x)
        mse_loss = criterion(reconstruction, x)
        kl_loss = kl_divergence(mu[:,-nb_salient_features:], log_var[:,-nb_salient_features:]) * kl_scaler

        f1, f2 = torch.split(torch.nn.functional.normalize(z[:,-nb_salient_features:]), [nb_x, nb_x], dim=0)

        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        contrastive_loss = sup_con_loss(features, y) * contrastive_scaler
        loss = mse_loss + kl_loss + contrastive_loss
        running_loss += loss.item()
        running_mse += mse_loss.item()
        running_kl += kl_loss.item()
        running_contrastive += contrastive_loss.item()
        loss.backward()
        optimizer.step()
        # print('contrastive_loss', contrastive_loss)
        # print('running_contrastive', running_contrastive)
    train_loss = running_loss / len(data_loader.dataset)
    train_mse = running_mse / len(data_loader.dataset)
    train_kl = running_kl / len(data_loader.dataset)
    train_contrastive = running_contrastive / len(data_loader.dataset)
    model.update_loss(
        loss=train_loss,
        mse=train_mse,
        kl=train_kl,
        contrastive=train_contrastive,
        train=True,
    )
    model.update_kl_scaler(kl_scaler)
    return train_loss, train_mse, train_kl, train_contrastive


def validate_salient_contrastive_nosalient_kl(
    model, criterion, sup_con_loss, data_loader, device, kl_scaler, contrastive_scaler, nb_salient_features
):
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_kl = 0.0
    running_contrastive = 0.0
    with torch.no_grad():
        for x, y in data_loader:
            y = y.argmax(axis=1).to(device)
            x = torch.cat([x[0].to(device), x[1].to(device)], dim=0)
            nb_x = y.shape[0]
            reconstruction, z, mu, log_var = model(x)
            mse_loss = criterion(reconstruction, x)
            kl_loss = kl_divergence(mu[:,-nb_salient_features:], log_var[:,-nb_salient_features:]) * kl_scaler
            f1, f2 = torch.split(torch.nn.functional.normalize(z[:,-nb_salient_features:]), [nb_x, nb_x], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            contrastive_loss = sup_con_loss(features, y) * contrastive_scaler
            loss = mse_loss + kl_loss + contrastive_loss
            running_loss += loss.item()
            running_mse += mse_loss.item()
            running_kl += kl_loss.item()
            running_contrastive += contrastive_loss.item()

    val_loss = running_loss / len(data_loader.dataset)
    val_mse = running_mse / len(data_loader.dataset)
    val_kl = running_kl / len(data_loader.dataset)
    val_contrastive = running_contrastive / len(data_loader.dataset)
    model.update_loss(
        loss=val_loss, mse=val_mse, kl=val_kl, contrastive=val_contrastive, train=False
    )
    return val_loss, val_mse, val_kl, val_contrastive


class DatasetTrainScaledContrastive(Dataset):
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
        x_ts_output = [
            np.empty((self.nb_variables, self.output_size)),
            np.empty((self.nb_variables, self.output_size)),
        ]
        for i in range(2):
            if self.transform:
                if np.random.rand() < self.scaling_prob:
                    size_scaled = np.random.randint(
                        int(self.output_size * self.scaling_inf),
                        int(self.output_size * self.scaling_max),
                    )
                    start_index_crop = np.random.randint(
                        0, self.input_size - size_scaled
                    )
                    x_ts = (
                        self.x[
                            index,
                            :,
                            start_index_crop : (start_index_crop + size_scaled),
                        ]
                        .detach()
                        .numpy()
                    )
                    
                    x_ts_output[i] = torch.tensor(scs.resample(x_ts, num=self.output_size, axis=1))
                else:
                    start_index_crop = np.random.randint(
                        0, self.input_size - self.output_size
                    )
                    # random crop
                    x_ts_output[i] = self.x[
                        index,
                        :,
                        start_index_crop : (start_index_crop + self.output_size),
                    ]
            else:
                # center crop
                start_index_crop = self.input_size // 2 - self.output_size // 2
                x_ts_output[i] = self.x[
                    index, :, start_index_crop : (start_index_crop + self.output_size)
                ]

        return x_ts_output, self.y[index]

    def __len__(self):
        return self.x.shape[0]
    
class CNN_ContrastiveVAE(VAE):
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
        super(CNN_ContrastiveVAE, self).__init__(scaler)

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
        return reconstruction, z, mu, log_var

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

    def update_loss(self, loss, mse, kl, contrastive, train=True):
        if train:
            self.train_loss = np.append(self.train_loss, loss)
            self.train_mse = np.append(self.train_mse, mse)
            self.train_kl = np.append(self.train_kl, kl)
            self.train_contrastive = np.append(self.train_contrastive, contrastive)
        else:
            self.test_loss = np.append(self.test_loss, loss)
            self.test_mse = np.append(self.test_mse, mse)
            self.test_kl = np.append(self.test_kl, kl)
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

        top_kl = (
            np.percentile(self.train_kl / self.kl_scaler, [50])[0]
            + np.percentile(self.train_kl / self.kl_scaler, [50])[0] / 2
        )
        min_kl = (
            np.percentile(self.train_kl / self.kl_scaler, [50])[0]
            - np.percentile(self.train_kl / self.kl_scaler, [50])[0] / 2
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
        plt.title("kl")
        plt.plot(self.train_kl / self.kl_scaler)
        plt.plot(self.test_kl / self.kl_scaler, ":", color="green")
        plt.ylim(top=top_kl, bottom=min_kl)
        plt.grid()
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
            reconstruction, _, _, _ = self.forward(
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

        
class TwoHeadedEncoderContrastiveVAE(VAE):
    def __init__(
        self,
        scaler,
        salient_dim,
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
        super(TwoHeadedEncoderContrastiveVAE, self).__init__(scaler)

        self.salient_dim = salient_dim
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
        
        
        self.enc_layer3_global = nn.Sequential(
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
        
        self.enc_layer3_salient = nn.Sequential(
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
        

        self.fc_mu_global = nn.Linear(
            in_features=self.bottleneck_length * self.n_filters_c3,
            out_features=self.latent_dim-self.salient_dim,
        )
        self.fc_var_global = nn.Linear(
            in_features=self.bottleneck_length * self.n_filters_c3,
            out_features=self.latent_dim-self.salient_dim,
        )
        
        self.fc_mu_salient = nn.Linear(
            in_features=self.bottleneck_length * self.n_filters_c3,
            out_features=self.salient_dim,
        )
        self.fc_var_salient = nn.Linear(
            in_features=self.bottleneck_length * self.n_filters_c3,
            out_features=self.salient_dim,
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
        return reconstruction, z, mu, log_var

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
        x_global = self.enc_layer3_global(x)
        x_salient = self.enc_layer3_salient(x)
        
        x_global = torch.flatten(x_global, start_dim=1)
        x_salient = torch.flatten(x_salient, start_dim=1)
        
        mu_global = self.fc_mu_global(x_global)
        log_var_global = self.fc_var_global(x_global)
        mu_salient = self.fc_mu_salient(x_salient)
        log_var_salient = self.fc_var_salient(x_salient)
        
        mu = torch.cat((mu_global, mu_salient), -1)
        log_var = torch.cat((log_var_global, log_var_salient), -1)
        return mu, log_var

    def update_loss(self, loss, mse, kl, contrastive, train=True):
        if train:
            self.train_loss = np.append(self.train_loss, loss)
            self.train_mse = np.append(self.train_mse, mse)
            self.train_kl = np.append(self.train_kl, kl)
            self.train_contrastive = np.append(self.train_contrastive, contrastive)
        else:
            self.test_loss = np.append(self.test_loss, loss)
            self.test_mse = np.append(self.test_mse, mse)
            self.test_kl = np.append(self.test_kl, kl)
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

        top_kl = (
            np.percentile(self.train_kl / self.kl_scaler, [50])[0]
            + np.percentile(self.train_kl / self.kl_scaler, [50])[0] / 2
        )
        min_kl = (
            np.percentile(self.train_kl / self.kl_scaler, [50])[0]
            - np.percentile(self.train_kl / self.kl_scaler, [50])[0] / 2
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
        plt.title("kl")
        plt.plot(self.train_kl / self.kl_scaler)
        plt.plot(self.test_kl / self.kl_scaler, ":", color="green")
        plt.ylim(top=top_kl, bottom=min_kl)
        plt.grid()
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
            reconstruction, _, _, _ = self.forward(
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

def get_vae_contrastive_latent(model, X, batch_size, device):
    # torch dataset generator
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X.astype(np.float32))),
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
            _, _, mu, _ = model(x)
            x_latent += [mu.to("cpu").detach().numpy()]

    x_latent = np.concatenate(x_latent, axis=0)
    return x_latent


def plot_embeddings(model, X_train, Y_train, X_test, Y_test, mlb, device, nb_salient_features=np.nan, supervised=False, path=""):
    
    t0 = time()
    latent_train = get_vae_contrastive_latent(
        model, X_train[:, :, 372:628], batch_size=128, device=device
    )
    latent_test = get_vae_contrastive_latent(
        model, X_test[:, :, 372:628], batch_size=128, device=device
    )
    if nb_salient_features != nb_salient_features :
        nb_salient_features = latent_train.shape[1]
        
    latent_train = latent_train[:, -nb_salient_features:]
    latent_test = latent_test[:, -nb_salient_features:]
    
    print("latent : ", time() - t0)

    umap_2d = umap.UMAP(random_state=0, n_neighbors=30, n_components=2, n_epochs=200)

    tu = time()
    u_embedding_train = umap_2d.fit_transform(latent_train)
    u_embedding_test = umap_2d.transform(latent_test)
    print("u embedd : ", time() - tu)

    nb_rows = 1
    if supervised : 
        ts = time()
        s_embedding_train = umap_2d.fit_transform(latent_train, y=Y_train.argmax(axis=1))
        s_embedding_test = umap_2d.transform(latent_test)
        print("s embedd : ", time() - ts)
        nb_rows = 2
        
    plt.figure(figsize=(20, 10*nb_rows))
    plt.subplot(nb_rows, 2, 1)
    plt.title("Train")
    for i, label in enumerate(mlb.classes_):
        idx_label = np.where(Y_train.argmax(axis=1) == i)[0]
        plt.plot(
            u_embedding_train[idx_label, 0],
            u_embedding_train[idx_label, 1],
            ".",
            alpha=0.5,
            label=label,
        )
    plt.legend()
    plt.ylabel("UNSUPERVISED")
    plt.grid()
    plt.subplot(nb_rows, 2, 2)
    plt.title("Test")
    for i, label in enumerate(mlb.classes_):
        idx_label = np.where(Y_test.argmax(axis=1) == i)[0]
        plt.plot(
            u_embedding_test[idx_label, 0],
            u_embedding_test[idx_label, 1],
            ".",
            alpha=0.5,
            label=label,
        )
    plt.legend()
    plt.grid()
    if supervised : 
        plt.subplot(nb_rows, 2, 3)
        for i, label in enumerate(mlb.classes_):
            idx_label = np.where(Y_train.argmax(axis=1) == i)[0]
            plt.plot(
                s_embedding_train[idx_label, 0],
                s_embedding_train[idx_label, 1],
                ".",
                alpha=0.5,
                label=label,
            )
        plt.legend()
        plt.ylabel("SUPERVISED")
        plt.grid()
        plt.subplot(nb_rows, 2, 4)
        for i, label in enumerate(mlb.classes_):
            idx_label = np.where(Y_test.argmax(axis=1) == i)[0]
            plt.plot(
                s_embedding_test[idx_label, 0],
                s_embedding_test[idx_label, 1],
                ".",
                alpha=0.5,
                label=label,
            )
        plt.legend()
        plt.grid()
    if path == "":
        plt.show()
    else:
        plt.savefig(path, dpi=150)
    plt.close("all")

    print("time total ", time() - t0)
    return umap_2d