import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import os
import sys
import time

from src.AutoIPQ import *
from src.utils import *
from src.simulate import *
from src.ML import *
from src.ForwardModel import *
from shared_utils import instantiate_sim, set_default_parameters, load_parameters_file, add_gaussian_noise_X, load_dataset, loss_func_handle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR




#######################################
# Setup
#######################################

project_path = "/scratch/arp/domain/"
total_instances = 200 # 200
budget = 20000
#budget = 100 # TODO: remove
overwrite = False

param_paths = {
    "prosail": "experiments/params_prosail.json",
    "prosail_2d": "experiments/params_prosail_2d.json",
    #"sbi_gm": "/home/arp/experiments/params_sbi_gm.json",
    #"sbi_bglm": "/home/arp/AutoIPQ/experiments/params_sbi_bglm.json",
}


    
loss_functions = ["proportional_distance", "sam"]
#loss_functions = ["sam"] # TODO: remove
#loss_functions = ["proportional_distance"]
datasets = ["simulated", "real"]
datasets = ["simulated"]
prosail_setups = ["prosail", "prosail_2d"]
#prosail_setups = ["prosail"]

conds_all = {}
i = 1
for lf in loss_functions:
    for ds in datasets:
        for ps in prosail_setups:
            conds_all[i] = {"loss_func":lf, "dataset":ds, "prosail_setup":ps}
            i += 1
        
conds = conds_all[int(sys.argv[1])]





# Set up some dirs

currdir = project_path
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass

currdir = project_path + "results/"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
        
currdir = currdir + conds["dataset"] + "/"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
        
currdir = currdir + conds["prosail_setup"] + "/"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
        
currdir = currdir + conds["loss_func"] + "/"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
            
currdir = currdir + "E5" + "/"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass



def load_data_synth(p):
    sim = instantiate_sim(p["simdata_simulator"])
    if(p["simdata_generator"] == "random"):
        data_generator = RandomLUTGenerator(sim, p["exp_parameters"], seed=p["exp_seed"], sim_parameters=p["exp_parameters_simulation"])
    elif(p["simdata_generator"] == "LHS"):
        data_generator = LHSLUTGenerator(sim, p["exp_parameters"], seed=p["exp_seed"], sim_parameters=p["exp_parameters_simulation"])
    else:
        data_generator = UniformLUTGenerator(sim, p["exp_parameters"], seed=p["exp_seed"], sim_parameters=p["exp_parameters_simulation"])

    X_synth, Y_synth_all, M_synth_table = data_generator.generate(p["simdata_num_instances"])
    X_train_synth, X_test_synth, Y_train_synth_all, Y_test_synth_all, M_train_synth_table, M_test_synth_table = train_test_split(X_synth, Y_synth_all, M_synth_table, test_size=p["data_test_prop"])

    M_synth = []
    M_train_synth = []
    M_test_synth = []

    for i in range(0, M_synth_table.shape[0]):
        d = {p["exp_parameters_simulation"][j]:M_synth_table[i, j] for j in range(len(p["exp_parameters_simulation"]))}
        d["platform"] = "sentinel-2a"
        M_synth.append(d)
    for i in range(0, M_train_synth_table.shape[0]):
        d = {p["exp_parameters_simulation"][j]:M_train_synth_table[i, j] for j in range(len(p["exp_parameters_simulation"]))}
        d["platform"] = "sentinel-2a"
        M_train_synth.append(d)
    for i in range(0, M_test_synth_table.shape[0]):
        d = {p["exp_parameters_simulation"][j]:M_test_synth_table[i, j] for j in range(len(p["exp_parameters_simulation"]))}
        d["platform"] = "sentinel-2a"
        M_test_synth.append(d)


    #print("\nSynthetic data test set size: ")
    #print("X: ", X_test_synth.shape)
    #print("Y: ", Y_test_synth.shape)

    #print("Min, max, mean, median:")
    #print(np.min(Y_test_synth), np.max(Y_test_synth), np.mean(Y_test_synth), np.median(Y_test_synth))
    
    data = {
        "X_train_synth": X_train_synth,
        "X_test_synth": X_test_synth,
        "Y_train_synth_all": Y_train_synth_all,
        "Y_test_synth_all": Y_test_synth_all,
        "M_train_synth": M_train_synth,
        "M_test_synth": M_test_synth,
    }
    
    return(data)


 
#######################################
# Running experiment
#######################################


# Setup
p = set_default_parameters()
load_parameters_file(p, param_paths[conds["prosail_setup"]])
p["exp_parameters_simulation"] = [] # Don't do varied geometry for simple check
p["data_test_prop"] = 0.2
sim = instantiate_sim("prosail")
param_ranges = sim.get_param_ranges()


#data = load_dataset(conds["prosail_setup"], "/scratch/arp/AutoIPQ/", max_instances=total_instances)
#X = data["test"]["X"]
#Y = data["test"]["Y"]
#M = data["test"]["M"]

# Data

sim_data1 = load_data_synth(p)
p["exp_seed"] = p["exp_seed"] + 23094
sim_data2 = load_data_synth(p)
p["exp_seed"] = p["exp_seed"] + 98475
sim_data3 = load_data_synth(p)

X_test1 = sim_data1["X_test_synth"]
X_test2 = sim_data2["X_test_synth"]
X_test3 = sim_data3["X_test_synth"]


Y_test1 = sim_data1["Y_test_synth_all"]
Y_test2 = sim_data2["Y_test_synth_all"]
Y_test3 = sim_data3["Y_test_synth_all"]

Y_test_mean = np.zeros_like(Y_test1)
Y_test_median = np.zeros_like(Y_test1)


# Combine 3 different spectral sources
X_test = np.zeros_like(X_test1)
for i in range(X_test.shape[0]):
    # Compute weighted sum for spectrum (sum to 1)
    w1 = np.random.random()
    w2 = np.random.random()
    w3 = np.random.random()
    tot = w1+w2+w3
    w1 = w1/tot
    w2 = w2/tot
    w3 = w3/tot
    X_test[i, :] = w1*X_test1[i, :] + w2*X_test2[i, :] + w3*X_test3[i, :]
    
    # Store meaningful quantities for Y
    Y_test_mean[i, :] = w1*Y_test1[i, :] + w2*Y_test2[i, :] + w3*Y_test3[i, :]
    Y_test_median[i, :] = np.median([Y_test1[i,:], Y_test2[i,:], Y_test3[i,:]], axis=0)
    




loss_matrices = []

# TODO: Loading existing instances
existing_instances = []
for d in os.listdir(currdir):
    d2 = d.split(".")[0]
    if(len(d2) < 5 and not(overwrite)):
        existing_instances.append(int(d2))
        
        mat = pd.read_csv(currdir + d).values[:,1:].astype(float)
        loss_matrices.append(mat)



# Iterate over instances
for i in range(min(X_test.shape[0], total_instances)): 
    if(overwrite or i not in existing_instances):
        # Save csv per instance; rows contain restarts
        
        # Header/overwrite stuff first
        csv_path = currdir + str(i) + ".csv"
        h = "parameter,loss_mean,loss_median,loss_Y1,loss_Y2,loss_Y3\n"
        if(not(os.path.exists(csv_path)) or overwrite):
            with open(csv_path, 'w') as fp:
                fp.write(h)

            # Setup priors
            priors = {}
            #if(len(M_test[i]) > 1):
            #    params_sim = p["exp_parameters_simulation"]
            #    priors = {params_sim[j]:M_test[i][j] for j in range(len(params_sim))}
            sim = instantiate_sim("prosail", priors=priors)
            param_ranges = sim.get_param_ranges()
            
            
            param_loss_matrix = np.zeros((len(p["exp_parameters"]), 5))
            
            
            instance_x = X_test[i, :]

            # Compute optimum
            method_kwargs = {
                                "budget": budget, 
                                "budget_split": [1.0, 0.0, 0.0],
                                "num_solutions": 2,
                                "return_opt": False, 
                                "store_XY": False, 
                                "store_test":False,
                                "train_clf": False,
                            }
                            
            auto = AutoIPQ(sim, instance_x, p["exp_parameters"], 2, optimiser=p["method_optimiser"], optimiser_2a=p["method_optimiser"], optimiser_2b=p["method_optimiser"], optimiser_2c=p["method_optimiser"], optimiser_2d=p["method_optimiser"], loss_func=conds["loss_func"])
            l_clf = auto.run(0.1, None, **method_kwargs)
            opt = auto.optima[0]
            opt_vec = auto.optimiser.solution_to_coords(opt) 

            for j in range(Y_test_mean.shape[1]):
                loss_mean = abs(abs(Y_test_mean[i, j]) - abs(opt_vec[j]))
                loss_median = abs(Y_test_median[i, j] - opt_vec[j])
                loss_Y1 = abs(Y_test1[i, j] - opt_vec[j])
                loss_Y2 = abs(Y_test2[i, j] - opt_vec[j])
                loss_Y3 = abs(Y_test3[i, j] - opt_vec[j])
            
                param_loss_matrix[j, :] = np.array([loss_mean, loss_median, loss_Y1, loss_Y2, loss_Y3])
            

            loss_matrices.append(param_loss_matrix)
                
            s = ""
            for i2 in range(len(p["exp_parameters"])):
                s = s + p["exp_parameters"][i2] + ","
                for j in range(param_loss_matrix.shape[1]):
                    s = s + str(param_loss_matrix[i2, j]) + ","
                s = s[:-1] + "\n"
            with open(csv_path, 'a') as fp:
                fp.write(s)
                
                    


# Aggregation: table, rows are parameters, columns are loss quantities (should probably normalise)

param_ranges = sim.get_param_ranges()
                    
# Compute mean loss for each Y quantity per parameter
param_perfs = np.zeros((len(p["exp_parameters"]), 5))
param_std_perfs = np.zeros((len(p["exp_parameters"]), 5))

for param_i in range(len(p["exp_parameters"])):
    quant_vec = np.zeros(loss_matrices[0].shape[1])
    quant_std_vec = np.zeros(loss_matrices[0].shape[1])
    for q_j in range(loss_matrices[0].shape[1]):
        vals = np.zeros(len(loss_matrices))
        for i in range(len(loss_matrices)):
            vals[i] = loss_matrices[i][param_i, q_j]
        # Also normalise here. New version: don't subtract min of range, because error is always starting at 0 (no need to shift)
        #quant_vec[q_j] = (np.mean(vals) - param_ranges[p["exp_parameters"][param_i]][2]) / (param_ranges[p["exp_parameters"][param_i]][3] - param_ranges[p["exp_parameters"][param_i]][2])
        quant_vec[q_j] = np.mean(vals) / (param_ranges[p["exp_parameters"][param_i]][3] - param_ranges[p["exp_parameters"][param_i]][2])
        #quant_vec[q_j] = np.mean(vals)
        quant_std_vec[q_j] = np.std(vals)  / (param_ranges[p["exp_parameters"][param_i]][3] - param_ranges[p["exp_parameters"][param_i]][2])
    param_perfs[param_i, :] = quant_vec
    param_std_perfs[param_i, :] = quant_std_vec
        
    
# Store table as csv
s = "parameter,loss_mean,loss_median,loss_Y1,loss_Y2,loss_Y3\n"
for i in range(param_perfs.shape[0]):
    s = s + p["exp_parameters"][i] + ","
    for j in range(param_perfs.shape[1]):
        s = s + "$" + str(np.round(param_perfs[i, j], 3)) + " \pm " + str(np.round(param_std_perfs[i, j], 3)) + "$,"
    s = s[:-1] + "\n"
with open(currdir+"E5_table.csv", 'w') as fp:
    fp.write(s)



print("Terminated successfully:", conds)