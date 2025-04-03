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
ci_sizes = [0.0, 0.1, 0.3, 0.5, 1.0] # 1.0 is equivalent to no prior
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
            
currdir = currdir + "E6" + "/"
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
X_test_true = sim_data["X_test_synth"]
Y_test = sim_data["Y_test_synth_all"]
M_test = sim_data["M_test_synth"]


    
X_test = X_test_true.copy()
for j in range(X_test_true.shape[1]):
    X_test[:, j] = X_test[:, j] + np.random.normal(loc=0, scale=0.1*np.mean(X_test_true[:, j]), size=X_test.shape[0])


loss_matrices = []

# Loading existing instances
existing_instances = []
for d in os.listdir(currdir):
    d2 = d.split(".")[0]
    if(len(d2) < 5 and not(overwrite)):
        existing_instances.append(int(d2))
        
        mat = pd.read_csv(currdir + d).values
        loss_matrices.append(mat)




lai_index = None
for j in range(len(p["exp_parameters"])):
    if(p["exp_parameters"][j] == 'lai'):
        lai_index = j
        break

# Iterate over instances
for i in range(min(X_test.shape[0], total_instances)): 
    if(overwrite or i not in existing_instances):
        csv_path = currdir + str(i) + ".csv"
        if(not(os.path.exists(csv_path)) or overwrite):

            # Save csv per instance; rows contain param+ci size combination
            
            # Header/overwrite stuff first
            h = "ci_size,lai_uniform,"
            for param in p["exp_parameters"]:
                h = h + param + ","
            h = h[:-1] + "\n"
            with open(csv_path, 'w') as fp:
                fp.write(h)

            # Setup priors
            priors = {}
            if(len(M_test[i]) > 1):
                params_sim = p["exp_parameters_simulation"]
                priors = {params_sim[j]:M_test[i][j] for j in range(len(params_sim))}
            sim = instantiate_sim("prosail", priors=priors)
            param_ranges = sim.get_param_ranges()
            
            
            rows = []
            
            result_matrix = np.zeros((len(ci_sizes), len(p["exp_parameters"])+1))
            
            # Iterate over noise levels
            ci_i = 0
            for ci in ci_sizes:
            
                instance_x = X_test[i, :]
                priors_local = {k:v for k,v in priors.items()}
                lai_true = Y_test[i, lai_index]

                params_local = p["exp_parameters"]
                lai_min = param_ranges['lai'][2]
                lai_max = param_ranges['lai'][3]
                total_size = lai_max - lai_min
                step = total_size * ci + 0.0000001 # smoothing to avoid problems if ci==0
                new_min = max(lai_min, lai_true-step)
                new_max = min(lai_max, lai_true+step)
                    
                sim_local = instantiate_sim("prosail", priors=priors)
                sim_local.set_ranges({'lai':['uniform', 'float', new_min, new_max]})
            
                # Start optimisation stuff
            
                # Set up AutoIPQ; only use step 1 result (optimum)
                method_kwargs = {
                                    "budget": budget, 
                                    "budget_split": [1.0, 0.0, 0.0],
                                    "num_solutions": 2,
                                    "return_opt": False, 
                                    "store_XY": False, 
                                    "store_test":False,
                                    "train_clf": False,
                                }
                                
                auto = AutoIPQ(sim_local, instance_x, params_local, 2, optimiser=p["method_optimiser"], optimiser_2a=p["method_optimiser"], optimiser_2b=p["method_optimiser"], optimiser_2c=p["method_optimiser"], optimiser_2d=p["method_optimiser"], loss_func=conds["loss_func"])
                l_clf = auto.run(0.1, None, **method_kwargs)
                opt = auto.optima[0]
                opt_vec = auto.optimiser.solution_to_coords(opt)
                # Compute loss
                for j in range(len(p["exp_parameters"])):
                    result_matrix[ci_i, j+1] = abs(opt[p["exp_parameters"][j]] - Y_test[i, j])
                result_matrix[ci_i, 0] = abs(np.random.uniform(new_min, new_max) - lai_true)
                                   
                ci_i += 1
                
            s = ""
            for i2 in range(len(ci_sizes)):
                s = s + str(ci_sizes[i2]) + ","
                for j in range(result_matrix.shape[1]):
                    s = s + str(result_matrix[i2, j]) + ","
                s = s[:-1] + "\n"
            with open(csv_path, 'a') as fp:
                fp.write(s)
                
                    
            loss_matrices.append(result_matrix)

                    
# Create table: rows are ci sizes, columns are average retrieval errors
aggr_table = np.zeros((len(ci_sizes), len(p["exp_parameters"])+1)) # Table to show
aggr_std_table = np.zeros((len(ci_sizes), len(p["exp_parameters"])+1)) # For adding standard deviations
aggr_all = np.zeros((len(ci_sizes), len(p["exp_parameters"])+1, len(loss_matrices))) # All points for significance

for ci_i in range(len(ci_sizes)):
    ci_vec = np.zeros(len(p["exp_parameters"])+1)
    ci_std_vec = np.zeros(len(p["exp_parameters"])+1)
    for j in range(1, len(p["exp_parameters"])+2):
        vals = np.zeros(len(loss_matrices))
        for i in range(len(loss_matrices)):
            vals[i] = loss_matrices[i][ci_i, j]
        ci_vec[j-1] = np.mean(vals)
        ci_std_vec[j-1] = np.std(vals)
        aggr_all[ci_i, j-1, :] = vals
    aggr_table[ci_i, :] = ci_vec
    aggr_std_table[ci_i, :] = ci_std_vec
    

# Significance test
from scipy.stats import wilcoxon
alpha = 0.05

aggr_sig_table = np.zeros_like(aggr_std_table)

for j in range(0, aggr_all.shape[1]):
    # Iterate over columns, for every row, count number of other rows being significantly outperformed
    for i in range(0, aggr_all.shape[0]):
        row_vals = aggr_all[i, j, :]
        num_outperformances = 1 # Technically now a rank instead of count of victories
        for i2 in range(0, aggr_all.shape[0]):
            if(not(i2 == i)):
                row_vals2 = aggr_all[i2, j, :]
                w = wilcoxon(row_vals, row_vals2, alternative='greater')
                if(w[1] < alpha):
                    num_outperformances += 1
                    
        aggr_sig_table[i, j] = num_outperformances



# Create table
    
s = "ci_size,lai_uniform,"
for param in p["exp_parameters"]:
    s = s + param + ","
s = s[:-1] + "\n"
for i in range(aggr_table.shape[0]):
    s = s + str(ci_sizes[i]) + ","
    for j in range(len(p["exp_parameters"])+1):
        s = s + "$[" + str(int(aggr_sig_table[i, j])) +"] " + str(np.round(aggr_table[i, j], 3)) + " \pm " + str(np.round(aggr_std_table[i, j], 3)) + "$,"
    s = s[:-1] + "\n"
with open(currdir+"E6_table.csv", 'w') as fp:
    fp.write(s)
    
print("aggr table", aggr_table)
aggr_table = aggr_table.T
print("aggr_table_new", aggr_table)
aggr_std_table = aggr_std_table.T
aggr_sig_table = aggr_sig_table.T

s = "parameter,0,10,30,50,100\n"
i = 0 
s = s + "lai_uniform,"
for j in range(aggr_table.shape[1]):
    s = s + "$[" + str(int(aggr_sig_table[i, j])) +"] " + str(np.round(aggr_table[i, j], 3)) + " \pm " + str(np.round(aggr_std_table[i, j], 3)) + "$,"
s = s[:-1] + "\n"
for i in range(1, aggr_table.shape[0]):
    s = s + p["exp_parameters"][i-1] + ","
    for j in range(aggr_table.shape[1]):
        s = s + "$[" + str(int(aggr_sig_table[i, j])) +"] " + str(np.round(aggr_table[i, j], 3)) + " \pm " + str(np.round(aggr_std_table[i, j], 3)) + "$,"
    s = s[:-1] + "\n"
with open(currdir+"E6_table_inv.csv", 'w') as fp:
    fp.write(s)
    


                
                


print("Terminated successfully:", conds)