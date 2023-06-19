import os
import sys
import json
from time import time
import datetime
import pickle

import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.vae_utils import CNN_VAE_3, fit, validate, DatasetTrainScaled, viz_reconstruction_multivariate, set_seeds
from data.utils import load_cured_data, split_ptb
mpl.use('Agg')

latent_dim = int(sys.argv[1])
random_seed = int(sys.argv[2])

dataset_path = "../../hot/ltsaot6/ptb"
sampling_rate = 100
ctype = "superdiagnostic"
test_fold = 10
length_ts = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss(reduction="mean")

training_params = {
    "batch_size": 256,
    "max_epochs": 1500,
    "scaling_param": 0.3,
    "kl_scaler": 0.0005,
    "contrastive_scaler": 0.1,
    "lr": 0.00005,
    "reduce_lr_patience": 45,
    "early_stopping": 100,
    "seed":random_seed,
}

model_params = {
    "input_channels": 12,
    "length_ts": 256,
    "n_filters_c1": 256,
    "size_filters_c1": 5,
    "n_filters_c2": 512,
    "size_filters_c2": 5,
    "n_filters_c3": 512,
    "size_filters_c3": 5,
    "latent_dim": latent_dim,
}

set_seeds(training_params["seed"], device)

model_name = f"vae_newkl_ld-{model_params['latent_dim']}_rs-{training_params['seed']}"
output_directory = f"../../data/ltsaot6/models/vae_new_kl/ld{latent_dim}/{model_name}"

if not os.path.exists(output_directory):
    os.mkdir(output_directory)

with open(f"{output_directory}/training_params.json", "w") as fp:
    json.dump(training_params, fp, sort_keys=True, indent=4)
with open(f"{output_directory}/model_params.json", "w") as fp:
    json.dump(model_params, fp, indent=4)

X, Y, target_y, mlb = load_cured_data(dataset_path, ctype)
X_train, Y_train, X_test, Y_test = split_ptb(X, Y, target_y, test_fold)
print(X.shape, Y.shape, target_y.shape)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    
model = CNN_VAE_3(scaler=StandardScaler(), **model_params)
model.to(device)
    
X_train, X_test = model.init_scaler(X_train=X_train, X_test=X_test, input_length=1000)
print(X_train.shape, X_test.shape)

# dataloaders scaled
train_dataset = DatasetTrainScaled(
    x=X_train,
    y=Y_train,
    transform=True,
    scaling_prob=1,
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
    scaling_prob=1,
    scaling_range=[
        1 - training_params["scaling_param"],
        1 + training_params["scaling_param"],
    ],
    output_size=model_params["length_ts"],
)

train_dataloader = DataLoader(
    train_dataset, batch_size=training_params["batch_size"], shuffle=True, num_workers=4
)
test_dataloader = DataLoader(
    test_dataset, batch_size=training_params["batch_size"], shuffle=False, num_workers=4
)

optimizer = optim.Adam(model.parameters(), lr=training_params["lr"])
scheduler = ReduceLROnPlateau(
    optimizer=optimizer, mode="min", patience=training_params["reduce_lr_patience"]
)

best_loss = 1e6
nb_stag = 0
nb_warmup = 25

warmup_lr = 0.00001
warmup_step = (training_params["lr"]-warmup_lr)/nb_warmup

t0_train = time()
for i in range(training_params["max_epochs"]):
    t0_epoch = time()
    if i < nb_warmup :
        for g in optimizer.param_groups:
            g['lr'] = warmup_lr
        warmup_lr += warmup_step

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
    time_epoch = time() - t0_epoch
    scheduler.step(test_loss)

    if test_loss < best_loss:
        best_loss = test_loss
        nb_stag = 0
        model.update_state_dict()
        viz_reconstruction_multivariate(
            model=model, x=X_test[0], device=device, path=f"{output_directory}/reconstruction_0.png"
        )
        viz_reconstruction_multivariate(
            model=model, x=X_test[1], device=device, path=f"{output_directory}/reconstruction_1.png"
        )
        model.viz_loss(path=f"{output_directory}/loss.png")

        with open(f"{output_directory}/model.pkl", "wb") as output:
            pickle.dump(model, output)
        print(
            f"Epoch {i}, Train loss {train_loss:.6f}, mse {train_mse:.6f}, kl {train_kl:.10f} // Test loss {test_loss:.6f}, mse {test_mse:.6f}, kl : {test_kl:.10f}; time : {int(time_epoch)}s, best epoch "
        )
    else:
        nb_stag += 1
        print(
            f"Epoch {i}, Train loss {train_loss:.6f}, mse {train_mse:.6f}, kl {train_kl:.10f} // Test loss {test_loss:.6f}, mse {test_mse:.6f}, kl : {test_kl:.10f}"
        )
    if nb_stag == training_params["early_stopping"]:
        break

time_train = time() - t0_train
print(str(datetime.timedelta(seconds=round(time_train))))

training_params['duration'] = time_train
with open(f"{output_directory}/training_params.json", "w") as fp:
    json.dump(training_params, fp, sort_keys=True, indent=4)


# save last model
with open(f"{output_directory}/model.pkl", "wb") as output:
    pickle.dump(model, output)
    
model.viz_loss(path=f"{output_directory}/loss.png")

#load best model
model.load_best()
best_epoch = model.test_loss.argmin()

viz_reconstruction_multivariate(
    model=model, x=X_test[0], device=device, path=f"{output_directory}/reconstruction_0.png"
)
viz_reconstruction_multivariate(
    model=model, x=X_test[1], device=device, path=f"{output_directory}/reconstruction_1.png"
)