import json
import os
import pickle
from glob import glob
from time import time 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import seaborn as sns
import torch
import umap.umap_ as umap
from tqdm import tqdm

from sklearn.utils import check_random_state
from sklearn.metrics import log_loss, accuracy_score
from sklearn.neighbors import KernelDensity

from sklearn.metrics import confusion_matrix, roc_auc_score
from src.vae_utils import get_reconstruction_error, get_validation_outputs
from src.eval_utils import * 
from data.utils import load_cured_data, split_ptb

model_path = "../../../data/ltsaot6/models"
dataset_path = "data/processed"
lime_path = 'outputs/lime'
list_contrastive_vae_64 = sorted(glob(f"{model_path}/contrastive_vae/ld64/*"))
list_contrastive_vae_32 = sorted(glob(f"{model_path}/contrastive_vae/ld32/*"))
list_contrastive_vae_16 = sorted(glob(f"{model_path}/contrastive_vae/ld16/*"))

list_vae_16 = sorted(glob(f"{model_path}/vae/ld16/*"))
list_vae_32 = sorted(glob(f"{model_path}/vae/ld32/*"))
list_vae_64 = sorted(glob(f"{model_path}/vae/ld64/*"))

list_contrastive_ae_16 = sorted(glob(f"{model_path}/contrastive_ae/ld16/*"))
list_contrastive_ae_32 = sorted(glob(f"{model_path}/contrastive_ae/ld32/*"))
list_contrastive_ae_64 = sorted(glob(f"{model_path}/contrastive_ae/ld64/*"))


sampling_rate = 100
ctype = "superdiagnostic"
test_fold = 10
length_ts = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_lime_norm = np.load(f'{lime_path}/lime_pred_test_norm.npy')

list_contrastive_vae = list_contrastive_vae_16 + list_contrastive_vae_32 +list_contrastive_vae_64
list_vae = list_vae_16+list_vae_32 +list_vae_64
list_contrastive_ae = list_contrastive_ae_16 +list_contrastive_ae_32 +list_contrastive_ae_64 
list_all_models = list_contrastive_vae + list_vae + list_contrastive_ae

print(len(list_all_models))

X, Y, target_y, mlb = load_cured_data(dataset_path, ctype)
x_train, y_train, x_test, y_test = split_ptb(X, Y, target_y, test_fold)
print(X.shape, Y.shape)
print(x_test.shape, y_test.shape)



sparsity_eps = .25

plot_tests = False
plot_wasserstein = False
plot_mean_preds = False
save_plots = False
verbose = False

list_res = []

baseline_model = pickle.load(
    open(f"{model_path}/CNN4-64-128-256-512-fc_512/model.pkl", "rb")
)
baseline_model = baseline_model.to(device)


for model_path in tqdm(list_all_models, position=0):
    res_dict = {}
    model_name = model_path.split("/")[-1]
    model_type = model_path.split("/")[6]

    with open(f"{model_path}/model_params.json", "r") as f:
        model_params = json.load(f)
    with open(f"{model_path}/training_params.json", "r") as f:
        training_params = json.load(f)
    latent_dim = model_params["latent_dim"]
    try:
        salient_dim = training_params["salient_dim"]
    except:
        salient_dim = 0

    try:
        seed = training_params["seed"]
    except:
        seed = -1
    try:
        train_time = training_params["duration"]
    except:
        train_time = np.nan
    
    gen_dim = latent_dim - salient_dim
    
    model = pickle.load(open(f"{model_path}/model.pkl", "rb"))
    model = model.to(device)
    model.load_best()
    
    
    x_train_scaled, x_test_scaled = model.rescale(x_train), model.rescale(x_test)
    mse = get_reconstruction_error(model, x_test, device, model_type)
    x_input_test, reconstruction_test, x_latent_test, y_list = get_validation_outputs(
        model, x_test_scaled, y_test, 128, device, model_type=model_type
    )
    
    
    res_dict.update({
        'seed':seed,
        'training_time':train_time,
        'model_path':model_path,
        'model_name':model_name,
        'type':model_type,
        'latent':latent_dim,
        'salient':salient_dim,
        'mse':mse,
    })

    # projection features
    if model_type == 'vae':
        dist_centroid = compute_centroid_dist(x_latent_test)
        mean_radius = compute_radius(x_latent_test)
        res_dict.update({
            'dist_centroid_z':dist_centroid,
            'mean_radius_z':mean_radius,
        })
        
    if model_type in ["contrastive_vae", "contrastive_ae"]:
        x_latent_train = get_code(model, x_train_scaled[:, :, 372:628], model_type = model_type)
        z_general, z_salient = x_latent_test[:,:gen_dim], x_latent_test[:,gen_dim:]

        dist_centroid_zg = compute_centroid_dist(z_general)
        dist_centroid_zs = compute_centroid_dist(z_salient)
        mean_radius_zg = compute_radius(z_general)
        mean_radius_zs = compute_radius(z_salient)
        res_dict.update({
            'dist_centroid_zg':dist_centroid_zg,
            'dist_centroid_zs':dist_centroid_zs,
            'mean_radius_zg':mean_radius_zg,
            'mean_radius_zs':mean_radius_zs
        })
        
        target, name = 3, 'NORM'
        (reconstruction_cf_max, reconstruction_cf_sub, reconstruction_cf_median), dict_time = project_ts_all(model, x_latent_train, x_latent_test, gen_dim, y_train, target=target)
        
        reconstruction_test_unscaled = model.rescale(reconstruction_test, inverse=True, input_length=256)
        x_input_test_unscaled = model.rescale(x_input_test, inverse=True, input_length=256)
        
        for rec_cf, cf_type in zip([reconstruction_cf_max, reconstruction_cf_sub, reconstruction_cf_median], ['max','sub','med']):
            df_res_mean, perfs_dict= get_cf_perfs(
                baseline_model,
                reconstruction_test_unscaled,
                model.rescale(rec_cf, inverse=True, input_length=256),
                x_input_test_unscaled,
                y_list,
                mlb,
                target=target,
                suffix=f'_{cf_type}_{name}'
            )
            
            orig_dist = get_distances(x_input_test, rec_cf, suffix=f'_{cf_type}_original_{name}') 
            rec_dist = get_distances(reconstruction_test, rec_cf, suffix=f'_{cf_type}_reconstruction_{name}') 
            res_dict.update(perfs_dict)
            res_dict.update(orig_dist)
            res_dict.update(rec_dist)
            res_dict.update(dict_time)
            
            mean_significant_diff, mean_small_diff = get_lime_cf_comparison(reconstruction_test, rec_cf, x_lime_norm, y_test)
            res_dict.update({
                f'mean_lime_changed_{cf_type}':mean_significant_diff,
                f'mean_lime_unchanged_{cf_type}':mean_small_diff,
            })
                
            sparsity_orig_cf = compute_sparsity(x_input_test, rec_cf, eps=sparsity_eps)
            sparsity_rec_cf = compute_sparsity(reconstruction_test, rec_cf, eps=sparsity_eps)
            res_dict.update({
                f'sparsity_orig_cf_{cf_type}_{name}':sparsity_orig_cf,
                f'sparsity_rec_cf_{cf_type}_{name}':sparsity_rec_cf,
            })
            
            df_res_mean.to_csv(f'outputs/mean_preds/df_res_mean_{model_name}_{cf_type}')
    
        
    list_res += [res_dict]


df_res = pd.DataFrame(list_res)
df_res.to_csv('df_res.csv', index=False)
print(df_res.shape)