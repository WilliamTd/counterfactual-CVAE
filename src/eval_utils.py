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

from data.utils import load_cured_data, split_ptb

def same_distrib_pval_1D(x0, x1, latent_dim):
    arr_cvm = np.zeros((7, latent_dim))
    arr_ks = np.zeros((7, latent_dim))
    for dec in range(7):
        for i in range(latent_dim):
            arr_cvm[dec, i] = sc.stats.cramervonmises_2samp(
                x0[dec::7, i], x1[dec::7, i]
            ).pvalue
            arr_ks[dec, i] = sc.stats.ks_2samp(x0[dec::7, i], x1[dec::7, i]).pvalue
    return arr_cvm, arr_ks


def wasserstein_1D(x0, x1):
    assert x0.shape[1] == x1.shape[1]
    latent_dim = x0.shape[1]
    arr_wasserstein = np.zeros((7, latent_dim))
    for dec in range(7):
        for i in range(latent_dim):
            arr_wasserstein[dec, i] = sc.stats.wasserstein_distance(
                x0[dec::7, i], x1[dec::7, i]
            )
    return arr_wasserstein.mean(axis=0)


def hsic_test(x, y):
    assert x.shape[0] == y.shape[0]
    hsic_stat = []
    hsic_pval = []
    hsic_thresh = []
    for dec in range(7):
        testStat, thresh, p_val = hsic_gam(x[dec::7], y[dec::7], alph=0.01)
        hsic_stat += [testStat]
        hsic_thresh += [thresh]
        hsic_pval += [p_val]
    return np.mean(hsic_stat), np.mean(hsic_thresh), np.mean(hsic_pval)


def get_code(model, x, model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    list_mu = []
    with torch.no_grad():
        model.eval()
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(x.astype(np.float32))),
            batch_size=128,
            shuffle=False,
        )
        for x in dataloader:
            x = x[0].to(device)
            if model_type in ["vae","contrastive_vae"]:
                mu, _ = model.encode(x)
            elif model_type in ["contrastive_ae"]:
                mu = model.encode(x)
            else :
                raise Exception(f'{model_type} not accepted as model_type')
            mu = mu.to("cpu").detach().numpy()
            list_mu += [mu]
    return np.concatenate(list_mu, axis=0)


def get_reconstruction_vae(model, x_latent):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    list_rec = []
    with torch.no_grad():
        model.eval()
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(x_latent.astype(np.float32))),
            batch_size=128,
            shuffle=False,
        )
        for x in dataloader:
            x = x[0].to(device)
            rec = model.decode(x)
            rec = rec.to("cpu").detach().numpy()
            list_rec += [rec]
    return np.concatenate(list_rec, axis=0)


def baseline_predict(baseline, x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_reshaped = baseline.rescale(x, input_length=x.shape[2])
    list_preds = []
    baseline = baseline.to(device)
    baseline.eval()
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(x_reshaped.astype(np.float32))),
            batch_size=128,
            shuffle=False,
        )
        for x in dataloader:
            x = x[0].to(device)
            preds = baseline(x)
            preds = torch.sigmoid(preds).to("cpu").detach().numpy()
            list_preds += [preds]
    return np.concatenate(list_preds, axis=0)


def get_kde_max_sub(x, x_other, kernel="epanechnikov", bandwidth=0.1):
    t_min = x.min(axis=0)
    t_max = x.max(axis=0)
    t = np.linspace(t_min, t_max, 1000)
    argmax = []
    for i in range(x.shape[1]):
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(
            np.reshape(x[:, i], (-1, 1))
        )
        kde_other = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(
            np.reshape(x_other[:, i], (-1, 1))
        )
        dens = np.exp(kde.score_samples(np.reshape(t[:, i], (-1, 1))))
        dens_other = np.exp(kde_other.score_samples(np.reshape(t[:, i], [-1, 1])))
        argmax += [t[(dens - dens_other).argmax(), i]]
    return np.asarray(argmax)


def get_kde_max(x, kernel="epanechnikov", bandwidth=0.1):
    t_min = x.min(axis=0)
    t_max = x.max(axis=0)
    t = np.linspace(t_min, t_max, 1000)
    argmax = []
    for i in range(x.shape[1]):
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(
            np.reshape(x[:, i], (-1, 1))
        )
        dens = np.exp(kde.score_samples(np.reshape(t[:, i], (-1, 1))))
        argmax += [t[dens.argmax(), i]]
    return np.asarray(argmax)


def project_ts(model, x_latent_train, x_latent_test, gen_dim, y_train, target):
    # where to project
    x_latent_train_salient = x_latent_train[:, gen_dim:]
    x_latent_test_salient = x_latent_test[:, gen_dim:]
    
    
    df_train = pd.DataFrame(x_latent_train_salient[np.where(y_train.argmax(axis=1)==target)[0],:2])
    
    x_min, x_max = np.percentile(df_train[0],[0.5,99.5])
    y_min, y_max = np.percentile(df_train[1],[0.5,99.5])
    sns.kdeplot(data=df_train,x=0, y=1)
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.show()
    
    x_latent_s_target = x_latent_train_salient[
        np.where(y_train.argmax(axis=1) == target)[0]
    ]
    target = get_kde_max(
        x_latent_s_target, kernel="epanechnikov", bandwidth=0.1
    )
    
    # latent space projection
    x_latent_test_proj = x_latent_test.copy()
    x_latent_test_proj[:, gen_dim:] = target
    # ts space projection
    reconstruction_proj = get_reconstruction_vae(model, x_latent_test_proj)
    return reconstruction_proj


def project_ts_all(model, x_latent_train, x_latent_test, gen_dim, y_train, target):
    # where to project
    x_latent_train_salient = x_latent_train[:, gen_dim:]
    x_latent_test_salient = x_latent_test[:, gen_dim:]
    
    t0 = time()
    x_latent_s_target = x_latent_train_salient[
        np.where(y_train.argmax(axis=1) == target)[0]
    ]
    t1 = time()
    x_latent_s_other = x_latent_train_salient[
        np.where(y_train.argmax(axis=1) != target)[0]
    ]
    t2 = time()
    
    t_latent_target = t1-t0
    t_latent_notarget = t2-t1
    
    t0 = time()
    target_mode = get_kde_max(
        x_latent_s_target, kernel="epanechnikov", bandwidth=0.1
    )
    t1 = time()
    target_sub = get_kde_max_sub(
        x_latent_s_target, x_latent_s_other, kernel="epanechnikov", bandwidth=0.1
    )
    t2 = time()
    target_median = np.median(x_latent_s_target, axis=0)
    t3 = time()
    
    t_kde_max = t1-t0
    t_kde_sub = t2-t1
    t_median = t3-t2
    
    
    list_proj = []
    list_time = []
    for tmp_target in [target_mode, target_sub, target_median]: 
        # latent space projection
        t0 = time()
        x_latent_test_proj = x_latent_test.copy()
        x_latent_test_proj[:, gen_dim:] = tmp_target
        # ts space projection
        list_proj += [get_reconstruction_vae(model, x_latent_test_proj)]
        t1 = time()
        
        list_time += [t1-t0]
    
    list_time[0] += t_kde_max
    list_time[1] += t_kde_sub
    list_time[2] += t_median
    
    list_time[0] /= x_latent_test.shape[0]
    list_time[1] /= x_latent_test.shape[0]
    list_time[2] /= x_latent_test.shape[0]

    dict_time = {
        f'inf_time_mode_{target}':list_time[0],
        f'inf_time_sub_{target}':list_time[1],
        f'inf_time_median_{target}':list_time[2],
    }
    return list_proj, dict_time


def get_distances(xa, xb, suffix=''):
    assert xa.shape==xb.shape
    diff  = xa- xb
    l1 = np.abs(diff).mean() 
    l2 = ((diff**2).sum(axis=2).sum(axis=1)/(diff.shape[1]*diff.shape[2])).mean()
    linf = np.abs(diff).max(axis=1).max(axis=1).mean() 
    if suffix != '':
        return {f'l1{suffix}':l1, f'l2{suffix}':l2, f'linf{suffix}':linf}
    return l1, l2, linf


def get_cf_perfs(
    baseline_model,
    reconstruction_test,
    reconstruction_cf,
    x_test_scaled,
    y_test,
    mlb,
    target,
    suffix='',
):
    idx_notarget = np.where(y_test.argmax(axis=1)!=target)[0]
    y_notarget = y_test[idx_notarget]

    
    preds_baseline_test = baseline_predict(baseline_model, x_test_scaled)
    preds_baseline_test_notarget = preds_baseline_test[idx_notarget]
    
    preds_baseline_test_reconstruction = baseline_predict(baseline_model, reconstruction_test)
    preds_baseline_test_reconstruction_notarget = preds_baseline_test_reconstruction[idx_notarget]
    
    preds_baseline_cf_reconstruction = baseline_predict(baseline_model, reconstruction_cf)
    preds_baseline_cf_reconstruction_notarget = preds_baseline_cf_reconstruction[idx_notarget]
    
    y_cf = np.zeros(y_notarget.shape)
    y_cf[:, target] = 1
    
    y_cf_all = np.zeros(y_test.shape)
    y_cf_all[:, target] = 1
    
    acc_rec_notarget = accuracy_score(
        y_true=y_notarget.argmax(axis=1),
        y_pred=preds_baseline_test_reconstruction_notarget.argmax(axis=1),
    )
    acc_cf_no_target = accuracy_score(
        y_true=y_cf.argmax(axis=1),
        y_pred=preds_baseline_cf_reconstruction_notarget.argmax(axis=1),
    )
    
    acc_cf_all = accuracy_score(
        y_true=y_cf_all.argmax(axis=1),
        y_pred=preds_baseline_cf_reconstruction.argmax(axis=1),
    )
    
    #with np.printoptions(precision=2):
    #    print(acc_baseline, acc_rec, acc_proj, acc_proj_all)
    #    print('reconstruction')
    #    print(confusion_matrix(y_true=y_test.argmax(axis=1), y_pred=preds_baseline_test_reconstruction.argmax(axis=1), normalize='true'))
    #    print('proj')
    #    print(confusion_matrix(y_true=y_test.argmax(axis=1), y_pred=preds_baseline_proj_reconstruction.argmax(axis=1), normalize='true'))

    list_original_label = []
    list_step = []
    list_preds = []
    for i, label in enumerate(mlb.classes_):
        idx = np.where(np.argmax(y_test, axis=1) == i)[0]
        pred_orig_mean = preds_baseline_test[idx].mean(axis=0)
        pred_rec_mean = preds_baseline_test_reconstruction[idx].mean(axis=0)
        pred_cf_mean = preds_baseline_cf_reconstruction[idx].mean(axis=0)
        list_original_label += [label, label, label]
        list_step += ["Original", "Reconstructed", "Counterfactual"]
        list_preds += [[pred_orig_mean], [pred_rec_mean], [pred_cf_mean]]
    preds = np.concatenate(list_preds)
    df_res_mean = pd.DataFrame(preds, columns=mlb.classes_)
    df_res_mean["original_label"] = list_original_label
    df_res_mean["step"] = list_step
    df_res_mean.set_index(["original_label", "step"], inplace=True)
    
    res_dict = {
        f'acc_rec_no_target{suffix}':acc_rec_notarget,
        f'acc_cf_no_target{suffix}':acc_cf_no_target,
        f'acc_cf_all{suffix}':acc_cf_all
    }
    
    return df_res_mean, res_dict


################### other cf metrics 

def compute_sparsity(ts_array1, ts_array2, eps=0.25):
    '''
    compute the % of the signal that did not significantly change
    '''
    assert ts_array1.shape==ts_array2.shape
    n = ts_array1.shape[0] * ts_array1.shape[1] * ts_array1.shape[2]
    return sum(sum(sum(np.abs(ts_array1-ts_array2)<eps)))/n


## CVAE metric separation 

def compute_centroid_dist(x_latent):
    n = int(x_latent.shape[0]/7)
    split_list = np.split(x_latent, n, axis=0)
    distance = []
    for split in split_list:
        centroid = split.mean(axis=0)
        distance += [np.mean(np.sqrt(sum((split-centroid)**2)))]
    return np.mean(distance)


def compute_radius(x_latent):
    n = int(x_latent.shape[0]/7)
    idx_x, idx_y = np.tril_indices(7)
    split_list = np.split(x_latent, n, axis=0)
    list_radius = []
    for split in split_list:
        radius = 0 
        for x,y in zip(idx_x,idx_y):
            if x!=y:
                d = np.sqrt(sum((split[x]-split[y])**2))
                if d>radius :
                    radius = d
        list_radius += [radius]
    return np.mean(list_radius)


def plot_ecg(X, legend='', path='', scaler=''):
    leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.title(leads[i])
        if scaler !='':
            plt.plot(np.linspace(0,2.56,256),(X[i]*scaler[1][i])+scaler[0][i], label=legend)
        else : 
            plt.plot(np.linspace(0,2.56,256),X[i], label=legend)
        if legend != '':
            plt.legend()
        plt.grid()
        
def get_lime_cf_comparison(x_rec, x_cf, ts_lime, y):
    err_cf = x_rec[3::7,:,:] - x_cf[3::7,:,:] 
    
    print(err_cf.shape, y.shape, ts_lime.shape)
    err_cf = err_cf[np.where(y.argmax(axis=1)!=3)]
    
    lime_big_r = ((np.abs(err_cf)>.25)*np.abs(ts_lime))
    lime_small_r = ((np.abs(err_cf)<.25)*np.abs(ts_lime))
    
    mean_significant_diff = np.abs(lime_big_r[np.where(lime_big_r!=0)]).mean()
    mean_small_diff = np.abs(lime_small_r[np.where(lime_small_r!=0.)]).mean()

    return mean_significant_diff, mean_small_diff


def get_lime_cf_comparison(x_rec, x_cf, ts_lime, y):
    err_cf = x_rec[3::7,:,:] - x_cf[3::7,:,:] 
    err_cf = err_cf[np.where(y.argmax(axis=1)!=3)]
    
    lime_big_r = ((np.abs(err_cf)>.25)*np.abs(ts_lime))
    lime_small_r = ((np.abs(err_cf)<.25)*np.abs(ts_lime))
    
    mean_significant_diff = np.abs(lime_big_r[np.where(lime_big_r!=0)]).mean()
    mean_small_diff = np.abs(lime_small_r[np.where(lime_small_r!=0.)]).mean()

    return mean_significant_diff, mean_small_diff