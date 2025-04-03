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
total_instances = 40 # 40
budget = 20000
#budget = 100 # TODO: remove
overwrite = True

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


pretty_names_parameters = {
    "lai": r"LAI",
    "cab": r"$C_{ab}$",
    "lidfa2": r"ALA",
    "cw": r"$C_{w}$",
    "psoil": r"soil moisture",
    "rsoil": r"soil brightness",
    "cm": r"$C_{m}$",
    "n": r"N",
    "car": r"$C_{ar}$",
    "cbrown": r"$C_{bp}$",
    "hspot": r"$hotspot$",
}



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
            
currdir = currdir + "E0" + "/"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
        
currdir = currdir + "convergence" + "/"
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




loss_matrices = []
spectral_loss_vecs = []

# TODO: Loading existing instances
existing_instances = []
for d in os.listdir(currdir):
    d2 = d.split(".")[0]
    if(len(d2) < 5 and not(overwrite)):
        existing_instances.append(int(d2))
        
        mat = pd.read_csv(currdir + d)['loss'].values
        loss_matrices.append(mat)



# Iterate over instances
for i in range(min(X_test.shape[0], total_instances)): 
    #if(overwrite or i not in existing_instances):
        #csv_path = currdir + str(i) + ".csv"
        #if(not(os.path.exists(csv_path)) or overwrite):

    # Save csv per instance; rows contain param+ci size combination
    
    # Header/overwrite stuff first
    #h = "parameter,"
    #for l in range(budget):
    #    h = h + "loss_" + str(l) + ","
    #h = h[:-1] + "\n"
    #with open(csv_path, 'w') as fp:
    #    fp.write(h)

    # Setup priors
    priors = {}
    if(len(M_test[i]) > 1):
        params_sim = p["exp_parameters_simulation"]
        priors = {params_sim[j]:M_test[i][j] for j in range(len(params_sim))}
    sim = instantiate_sim("prosail", priors=priors)
    param_ranges = sim.get_param_ranges()
    

    # Start optimisation stuff
    
    instance_x = X_test[i, :]

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
                    
    auto = AutoIPQ(sim, instance_x, p["exp_parameters"], 2, optimiser=p["method_optimiser"], optimiser_2a=p["method_optimiser"], optimiser_2b=p["method_optimiser"], optimiser_2c=p["method_optimiser"], optimiser_2d=p["method_optimiser"], loss_func=conds["loss_func"])
    l_clf = auto.run(0.1, None, track_convergence=True, **method_kwargs)
    
    Y_configs = auto.opt_local.get_evals("X_best") # x is iter, j is param
    loss_vec_spectrum = auto.opt_local.get_evals("L_best")
    
    # Parameter loss
    loss_vecs = {}    
    for j in range(Y_configs.shape[1]):
        param_vec = Y_configs[:, j]
        loss_vec = np.absolute(param_vec - Y_test[i, j])
        loss_vecs[p["exp_parameters"][j]] = loss_vec
        
    # Spectral loss
    #loss_vec_spectrum = np.zeros(Y_configs.shape[0])
    #for b in range(Y_configs.shape[0]):
    #    param_vec = Y_configs[b, :]
    #    sim_out = sim.simulate(auto.opt_local.coords_to_solution(param_vec))
    #    loss = np.mean(np.absolute(sim_out-instance_x)/np.absolute(instance_x))
    #    loss_vec_spectrum[b] = loss
        
    #losses = auto.opt_local.get_evals(eval_type='L')
    #df = pd.DataFrame.from_dict(loss_vecs, orient='index')
    
    #df.to_csv(csv_path)
            
    loss_matrices.append(loss_vecs)
    spectral_loss_vecs.append(loss_vec_spectrum)

                    
# Create convergence plot for parameters
for param in p["exp_parameters"]:
    mean_vec = np.zeros(budget)
    for d in loss_matrices:
        plt.plot(np.arange(len(d[param])), d[param], color='gray')
        mean_vec = mean_vec + d[param]
    
    mean_vec = mean_vec / len(loss_matrices)
    plt.plot(np.arange(len(mean_vec)), mean_vec, color='red')
    pretty_name = str(pretty_names_parameters[param])
    plt.title(r"Convergence to optimum: " + pretty_name)
    plt.xlabel("Iteration")
    plt.ylabel("Absolute error")
    #plt.yscale("log")
    plt.savefig(currdir+"convergence_"+str(param)+".pdf", bbox_inches='tight')
    plt.show()
    plt.clf()
    
    
# Convergence plot for spectrum
mean_vec = np.zeros(budget)
for loss_vec in spectral_loss_vecs:
    plt.plot(np.arange(len(loss_vec)), loss_vec, color='gray')
    mean_vec = mean_vec + loss_vec

mean_vec = mean_vec / len(spectral_loss_vecs)
plt.plot(np.arange(len(mean_vec)), mean_vec, color='red')
plt.title("Convergence to optimum: spectrum")
plt.xlabel("Iteration")
plt.ylabel("Spectral loss (log scale)")
plt.yscale("log")
plt.savefig(currdir+"convergence_spectrum.pdf", bbox_inches='tight')
plt.show()


print("Terminated successfully:", conds)