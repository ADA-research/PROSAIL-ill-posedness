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
num_restarts = 5
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
            
currdir = currdir + "E2" + "/"
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
sim_data = load_data_synth(p)

X_train = sim_data["X_train_synth"]
Y_train = sim_data["Y_train_synth_all"]
M_train = sim_data["M_train_synth"]
X_test = sim_data["X_test_synth"]
Y_test = sim_data["Y_test_synth_all"]
M_test = sim_data["M_test_synth"]



instance_list = []

existing_instances = []
for d in os.listdir(currdir):
    d2 = d.split(".")[0]
    if(len(d2) < 5):
        existing_instances.append(int(d2))
        
        opts = pd.read_csv(currdir + d).values
        instance_list.append(opts)

# Iterate over instances
for i in range(min(X_test.shape[0], total_instances)): 
    if(overwrite or i not in existing_instances):
        # Save csv per instance; rows contain restarts
        
        # Header/overwrite stuff first
        csv_path = currdir + str(i) + ".csv"
        h = ""
        for param in p["exp_parameters"]:
            h = h + param + ","
        h = h[:-1] + "\n"
        if(not(os.path.exists(csv_path)) or overwrite):
            with open(csv_path, 'w') as fp:
                fp.write(h)

            # Setup priors
            priors = {}
            if(len(M_test[i]) > 1):
                params_sim = p["exp_parameters_simulation"]
                priors = {params_sim[j]:M_test[i][j] for j in range(len(params_sim))}
            sim = instantiate_sim("prosail", priors=priors)
            param_ranges = sim.get_param_ranges()
            
            # Start restart stuff
            opts = np.zeros((num_restarts, len(p["exp_parameters"])))
            for i2 in range(num_restarts):
                
                # Randomly initialise (force uniform sampling)
                init_config = {param:np.random.uniform(param_ranges[param][2], param_ranges[param][3]) for param in p["exp_parameters"]}
            
                # Set up AutoIPQ; only use step 1 result (optimum)
                method_kwargs = {
                                    "budget": budget, 
                                    "budget_split": [1.0, 0.0, 0.0],
                                    "num_solutions": 2,
                                    "init": "manual",
                                    "init_params":init_config,
                                    "return_opt": False, 
                                    "store_XY": False, 
                                    "store_test":False,
                                    "train_clf": False,
                                }
                                
                instance_x = X_test[i, :]
                auto = AutoIPQ(sim, instance_x, p["exp_parameters"], 2, optimiser=p["method_optimiser"], optimiser_2a=p["method_optimiser"], optimiser_2b=p["method_optimiser"], optimiser_2c=p["method_optimiser"], optimiser_2d=p["method_optimiser"], loss_func=conds["loss_func"])
                l_clf = auto.run(0.1, None, **method_kwargs)
                opt = auto.optima[0]
                opt_vec = auto.optimiser.solution_to_coords(opt)
                opts[i2, :] = opt_vec
                
                # Add to file
                s = ""
                for j in range(len(opt_vec)):
                    s = s + str(opt_vec[j]) + ","
                s = s[:-1] + "\n"
                with open(csv_path, 'a') as fp:
                    fp.write(s)
                    
            instance_list.append(opts)

                    
# Normalise opts
for opts in instance_list:
    for j in range(len(p["exp_parameters"])):
        j_min = param_ranges[p["exp_parameters"][j]][2]
        j_max = param_ranges[p["exp_parameters"][j]][3]
        opts[:, j] = (opts[:, j] - j_min) / (j_max - j_min)

    
    
# Create histogram
vals = []
for inst in instance_list:
    max_dist = -1
    for i2 in range(inst.shape[0]):
        row = inst[i2, :]
        dists = np.mean(np.absolute(inst - row), axis=1)
        if(np.max(dists) > max_dist):
            max_dist = np.max(dists)
            
    vals.append(max_dist)
    

num_bins = 200
bins = np.arange(num_bins)/num_bins

a = plt.hist(vals, bins=bins, density=True, color='blue', label="Frequency")
count = a[0]
bins_count = a[1]
pdf = count / sum(count)
cdf = np.cumsum(pdf) * np.sum(count)
plt.plot(bins_count[1:], cdf, label="Cumulative frequency", color='orange', marker=".")
#plt.plot(bins, bins_count, label="CDF", color='orange')

plt.xlabel("Maximum distance")
plt.ylabel("Frequency")
plt.xlim(0, 1)
plt.title("Frequency of maximal distances between optima")
plt.legend()
plt.savefig(currdir+"histogram.pdf", bbox_inches='tight')
plt.show()
plt.clf()


a = plt.hist(vals, bins=bins, density=True, color='blue', label="Frequency")
count = a[0]
bins_count = a[1]
pdf = count / sum(count)
cdf = np.cumsum(pdf) * np.sum(count)
plt.plot(bins_count[1:], cdf, label="Cumulative frequency", color='orange', marker=".")

plt.xlabel("Maximum distance (log scale)")
plt.ylabel("Frequency")
plt.xlim(0.01, 1)
#plt.ylim(0, 20) # Not always correct, just to make the plot better
plt.xscale("log")
plt.title("Frequency of maximal distances between optima")
plt.legend()
plt.savefig(currdir+"histogram_log.pdf", bbox_inches='tight')
plt.show()



print("Terminated successfully:", conds)