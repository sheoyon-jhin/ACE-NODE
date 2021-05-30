from matplotlib import cm
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import gru_ode_bayes
from torch.utils.data import DataLoader
import gru_ode_bayes.data_utils as data_utils
import warnings
warnings.filterwarnings(action='ignore')
from gru_ode_bayes.datasets.double_OU import double_OU
import random
import matplotlib
import csv
matplotlib.use('agg')
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#import sys; sys.argv.extend(["--model_name", "paper_random_r", "--random_r"])
#import sys; sys.argv.extend(["--model_name", "BXLator", "--data_type", "BXLator","--format","pdf"])


def plot_trained_model(model_name="USHCN_original_2022", format_image="pdf", random_r=True, max_lag=0, jitter=0, random_theta=False, data_type="Climate"):

    style = "fill"
    model_name = "USHCN_original_234"
    model_name2 = "USHCN_new2_attention_L2_lambda0.1_234_0.25_0.0001_25_20"
    # model_name2 = "real_final_small_climate_attention_0.25_0.0001_25_20"
    
    
   
    summary_dict = np.load(f"/home/bigdyl/gru_ode_bayes/experiments/Climate/trained_models/{model_name}_params.npy", allow_pickle=True).item()
    summary_dict2 = np.load(f"/home/bigdyl/gru_ode_bayes/experiments/Climate/trained_models_atten/{model_name2}_params.npy",allow_pickle=True).item()

    
    params_dict = summary_dict
    metadata = summary_dict
    params_dict2 = summary_dict2
    metadata2 = summary_dict2
    if type(params_dict) == np.ndarray:
        ## converting np array to dictionary:
        params_dict = params_dict.tolist()

    #Loading model
    model = gru_ode_bayes.NNFOwithBayesianJumps(input_size=params_dict["input_size"], hidden_size=params_dict["hidden_size"],
                                                           p_hidden=params_dict["p_hidden"], prep_hidden=params_dict["prep_hidden"],
                                                           logvar=params_dict["logvar"], mixing=params_dict["mixing"],
                                                           classification_hidden=params_dict["classification_hidden"],
                                                           cov_size=params_dict["cov_size"], cov_hidden=params_dict["cov_hidden"],
                                                           dropout_rate=params_dict["dropout_rate"], full_gru_ode=params_dict["full_gru_ode"], impute=params_dict["impute"], store_hist=True)
                                                        #    batch=params_dict["batch"],store_hist=True)
    model2 = gru_ode_bayes.NNFOwithBayesianJumpsAttention(input_size=params_dict2["input_size"], hidden_size=params_dict2["hidden_size"],
                                                           p_hidden=params_dict2["p_hidden"], prep_hidden=params_dict2["prep_hidden"],
                                                           logvar=params_dict2["logvar"], mixing=params_dict2["mixing"],
                                                           classification_hidden=params_dict2["classification_hidden"],
                                                           cov_size=params_dict2["cov_size"], cov_hidden=params_dict2["cov_hidden"],
                                                           dropout_rate=params_dict2["dropout_rate"], full_gru_ode=params_dict2["full_gru_ode"], impute=params_dict2["impute"],
                                                           batch=params_dict2["batch"],store_hist=True)
   
    model.load_state_dict(torch.load(f"/home/bigdyl/gru_ode_bayes/experiments/Climate/trained_models/{model_name}_MAX.pt"),strict=False)
    model2.load_state_dict(torch.load(f"/home/bigdyl/gru_ode_bayes/experiments/Climate/trained_models_atten/{model_name2}_MAX.pt"),strict=False)
    model.eval()
    model2.eval()
    #Test data :
    N = 10
    T = metadata["T"]
    delta_t = metadata["delta_t"]
    
    # max_lag = metadata.pop("max_lag", None)
    
    csv_file_path = summary_dict["csv_file_path"]
    csv_file_cov = summary_dict["csv_file_cov"]
    csv_file_tags = summary_dict["csv_file_tags"]
    # import pdb
    # pdb.set_trace()
    test_idx = np.load("/home/bigdyl/gru_ode_bayes/gru_ode_bayes/datasets/Climate/folds/small_chunk_fold_idx_0/test_idx.npy",allow_pickle=True)
    # import pdb
    # pdb.set_trace()
    validation = True
    val_options = {"T_val": params_dict["T_val"], "max_val_samples": params_dict["max_val_samples"]}
    data_test = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags,
                                        cov_file= csv_file_cov, idx=test_idx, validation = validation,
                                        val_options = val_options)
    dl = DataLoader(dataset=data_test, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size=len(data_test))
    
    with torch.no_grad():
        for i, b in enumerate(dl):
            times = b["times"]
            time_ptr = b["time_ptr"]
            X = b["X"]
            M = b["M"]
            obs_idx = b["obs_idx"]
            cov = b["cov"]
            labels   = b["y"]
            batch_size = labels.size(0)
            if b["X_val"] is not None:
                X_val     = b["X_val"]
                M_val     = b["M_val"]
                times_val = b["times_val"]
                times_idx = b["index_val"]

            hT, loss, _, t_vec, p_vec, _, eval_times, eval_vals = model(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"], T=params_dict["T"], cov=cov, return_path=True)
            
            hT2, loss2, _, t_vec2, p_vec2, _, eval_times2, eval_vals2 = model2(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"], T=params_dict["T"], cov=cov, return_path=True,plot=True)
            
            observations=X.detach().numpy()
            mask = M.detach().numpy()
            t_vec = np.around(t_vec,str(params_dict["delta_t"])[::-1].find('.')).astype(np.float32) #Round floating points error in the time vector.
            t_vec2 = np.around(t_vec2,str(params_dict2["delta_t"])[::-1].find('.')).astype(np.float32)
            p_val = data_utils.extract_from_path(t_vec,p_vec,times_val,times_idx)
            p_val2 = data_utils.extract_from_path(t_vec2,p_vec2,times_val,times_idx)
                
            m, v = torch.chunk(p_val,2,dim=1)
            
            m2, v2 = torch.chunk(p_val2,2,dim=1)
            

            mse_at_list = []
            time_list = []
            mse_org_list = []
            for i in range(82):
                
                m2_10 = m2[(4*i):4*(i+1),:]
                m_10 = m[(4*i):4*(i+1),:]
                ob_10 = X_val[(4*i):4*(i+1),:]
                
                mask_10 = M_val[(4*i):4*(i+1),:]
                
                m_2 = ob_10 - m2_10 
                m_ = ob_10 - m_10 
                mse_at = (torch.square(m_2)*mask_10).mean()
                mse_org =(torch.square(m_)*mask_10).mean()
                
                mse_at = mse_at.item()
                mse_org = mse_org.item()
                mse_at_list.append(mse_at)
                mse_org_list.append(mse_org)
                time_list.append(i*4)
                
        plt.plot(time_list, mse_org_list, label = f"GRU-ODE-Bayes",color="orange")
        plt.plot(time_list, mse_at_list, label = f"Attentive\nGRU-ODE-Bayes",color="blue")
        plt.legend(loc="upper right",fontsize=20)
        plt.xlabel("Testing Time",fontsize=20)
        plt.ylabel("MSE",fontsize=20)
        fname = f"final_GRU-ODE-Bayes_seed_234.{format_image}"
        
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()



if __name__ =="__main__":
    plot_trained_model()
    manualSeed = 2020

    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)