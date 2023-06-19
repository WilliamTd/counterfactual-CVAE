import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import copy

class BaseModel(nn.Module):
    def __init__(self, scaler):
        super(BaseModel, self).__init__()

        self.train_loss = np.array([])
        self.test_loss = np.array([])
        self.train_auc = np.array([])
        self.test_auc = np.array([])
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

    def update_loss(self, loss, auc, train=True):
        if train:
            self.train_loss = np.append(self.train_loss, loss)
            self.train_auc = np.append(self.train_auc, auc)
        else:
            self.test_loss = np.append(self.test_loss, loss)
            self.test_auc = np.append(self.test_auc, auc)

    def viz_loss(self, path=""):
        plt.figure(figsize=(15,10))
        plt.subplot(1,2,1)
        plt.title(f"Loss, Epoch {len(self.train_loss)}")
        plt.plot(self.train_loss, label='Train')
        plt.plot(self.test_loss, ":", color="green", label="Test")
        plt.plot(np.minimum.accumulate(self.test_loss), color="green", label="Test min")
        plt.plot(np.argmin(self.test_loss), min(self.test_loss), "rx")
        plt.grid()
        plt.legend()
        plt.xlabel("Epochs")
        
        plt.subplot(1,2,2)
        plt.title('auc')
        plt.plot(self.train_auc, label='Train')
        plt.plot(self.test_auc, ":", color="green", label="Test")
        plt.plot(np.maximum.accumulate(self.test_auc), color="green", label="Test max")
        plt.plot(np.argmax(self.test_auc), max(self.test_auc), "rx")
        plt.grid()
        plt.legend()
        
        if path == "":
            plt.show()
        else:
            plt.savefig(path, dpi=150)

            
class CNN_3(BaseModel):
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
        n_fc,
        n_outputs,
    ):
        super(CNN_3, self).__init__(scaler)

        self.input_channels = input_channels
        self.length_ts = length_ts
        self.n_filters_c1 = n_filters_c1
        self.size_filters_c1 = size_filters_c1
        self.n_filters_c2 = n_filters_c2
        self.size_filters_c2 = size_filters_c2
        self.n_filters_c3 = n_filters_c3
        self.size_filters_c3 = size_filters_c3
        self.n_fc = n_fc
        self.n_outputs = n_outputs

        self.bottleneck_length = self.length_ts // 2 ** 2

        # encoder
        self.cnn_layer1 = nn.Sequential(
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
        self.cnn_layer2 = nn.Sequential(
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
        self.cnn_layer3 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_filters_c2,
                out_channels=self.n_filters_c3,
                kernel_size=self.size_filters_c3,
                padding=self.size_filters_c3 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c3),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(
                in_features=self.bottleneck_length * self.n_filters_c3,
                out_features=self.n_fc,
            ),
            nn.Dropout(p=0.5),
            nn.Linear(
                in_features=self.n_fc,
                out_features=self.n_outputs,
            ) 
        )

        
    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_out(x)
        return x
    

class CNN_4(BaseModel):
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
        n_fc,
        n_outputs,
    ):
        super(CNN_4, self).__init__(scaler)

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
        self.n_fc = n_fc
        self.n_outputs = n_outputs

        self.bottleneck_length = self.length_ts // 2 ** 3

        # encoder
        self.cnn_layer1 = nn.Sequential(
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
        self.cnn_layer2 = nn.Sequential(
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
        self.cnn_layer3 = nn.Sequential(
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
        self.cnn_layer4 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_filters_c3,
                out_channels=self.n_filters_c4,
                kernel_size=self.size_filters_c4,
                padding=self.size_filters_c4 // 2,
                padding_mode="replicate",
            ),
            nn.BatchNorm1d(self.n_filters_c4),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(
                in_features=self.bottleneck_length * self.n_filters_c4,
                out_features=self.n_fc,
            ),
            nn.Dropout(p=0.5),
            nn.Linear(
                in_features=self.n_fc,
                out_features=self.n_outputs,
            ) 
        )

        
    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = self.cnn_layer4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_out(x)
        return x