

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm 

from matplotlib.colors import LogNorm
from scipy.interpolate import griddata

from sklearn.model_selection import ParameterGrid, train_test_split

from fanova import fANOVA
import ConfigSpace
from ConfigSpace import UniformFloatHyperparameter

from src.AutoRTML import *
from src.utils import *
from src.simulate import *
from src.ML import *
from src.ForwardModel import *
from src.load_data import keys_to_numpy, get_keys_and_indices, remove_invalid

from general import *



#loss_funcs = ["SAM", "spectral_mae", "smae_b1", "smae_b2", "smae_b3", "smae_b4", "smae_b5", "smae_b6", "smae_b7", "smae_b8", "smae_b9", "smae_b10", "smae_b11",] 
loss_funcs = ["proportional_spectral_mae", "SAM"] 
save_fig = True


p = set_default_parameters()
p["exp_parameters"] = ["lai", "cab", "lidfa2", "psoil", "rsoil", 'cw', 'cm', 'n', 'car', 'cbrown', 'hspot', 'ant']


pretty_names_parameters = {
    "lai": r"LAI",
    "cab": r"$C_{ab}$",
    "lidfa2": r"ALA",
    "cw": r"$C_{w}$",
    "psoil": r"$G_{m}$",
    "rsoil": r"$G_{b}$",
    "cm": r"$C_{m}$",
    "n": r"N",
    "car": r"$C_{ar}$",
    "cbrown": r"$C_{brown}$",
    "ant": r"$C_{anth}$",
    "hspot": r"$hotspot$",
}

pretty_names_func = {
    "proportional_spectral_mae": "PMAE",
    "SAM": "SAM",
}


num_instances = p["simdata_num_instances"]
#num_instances = 5 # TODO: remove (for faster updates to plot code)



# Generating new data with all parameters
sim = instantiate_sim(p["simdata_simulator"])
if(p["simdata_generator"] == "random"):
    data_generator = RandomLUTGenerator(sim, p["exp_parameters"], seed=p["exp_seed"], sim_parameters=p["exp_parameters_simulation"])
elif(p["simdata_generator"] == "LHS"):
    data_generator = LHSLUTGenerator(sim, p["exp_parameters"], seed=p["exp_seed"], sim_parameters=p["exp_parameters_simulation"])
else:
    data_generator = UniformLUTGenerator(sim, p["exp_parameters"], seed=p["exp_seed"], sim_parameters=p["exp_parameters_simulation"])

X_synth2, Y_synth_all2, M_synth_table2 = data_generator.generate(num_instances)
M_synth2 = []

for i in range(0, M_synth_table2.shape[0]):
    d = {p["exp_parameters_simulation"][j]:M_synth_table2[i, j] for j in range(len(p["exp_parameters_simulation"]))}
    d["platform"] = "sentinel-2a"
    M_synth2.append(d)


for loss_func in loss_funcs:
    importances = np.zeros((num_instances, len(p["exp_parameters"])))
    for i in range(0, num_instances):

        m = M_synth2[i]
        priors_i = {}
        for param in p["exp_parameters_simulation"]:
            priors_i[param] = m[param]
        
        
        s = instantiate_sim(p["simdata_simulator"], priors=priors_i)

        X_config, y_loss = random_search(s, X_synth2[i, :], free_parameters=p["exp_parameters"], num_configs=200, loss_func=loss_func_handle(loss_func))

        # Experimental
        import ConfigSpace
        from ConfigSpace import UniformFloatHyperparameter
        config_space = ConfigSpace.ConfigurationSpace()
        ranges = s.get_param_ranges()
        k = 0 
        for param in p["exp_parameters"]:
            mn = ranges[param][2]
            mx = ranges[param][3]
            config_space.add(UniformFloatHyperparameter("x_%03i" % k, mn, mx))
            k += 1
        f = fANOVA(X_config, y_loss, config_space=config_space)

        #f = fANOVA(X_config, y_loss)

        imp_instance = [f.quantify_importance((j, ))[(j,)]['individual importance'] for j in range(0, len(p["exp_parameters"]))]
        importances[i, :] = imp_instance




    imps_dict = {}
    imps_std_dict = {}
    j = 0
    for param in p["exp_parameters"]:
        #print(param)
        #print(np.mean(importances[:, j]))
        #print("\n")
        imps_dict[param] = np.mean(importances[:, j])
        imps_std_dict[param] = np.std(importances[:, j])
        j += 1


    # Plotting
    keys_ordered = []
    keys_ordered_pretty = []
    for k in sorted(imps_dict, key=imps_dict.get, reverse=True):
        keys_ordered.append(k)
        keys_ordered_pretty.append(pretty_names_parameters[k])

    fig, ax = plt.subplots(figsize=(16,8))
    bar_x = np.arange(len(keys_ordered))
    bar_y = [imps_dict[k] for k in keys_ordered]
    bar_err = [imps_std_dict[k] for k in keys_ordered]
    ax.bar(bar_x, bar_y, yerr=bar_err)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(keys_ordered_pretty)
    ax.set_title("Parameter importance in PROSAIL simulations (" + pretty_names_func[str(loss_func)] + ")")
    ax.set_ylabel("Importance")
    ax.set_xlabel("Parameter")

    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    for a in ax.get_xticklabels():
        a.set_fontsize(20)
    for a in ax.get_yticklabels():
        a.set_fontsize(20)


    if(save_fig):
        plt.savefig("/scratch/arp/domain/results/" + "parameter_importance_" + str(loss_func) + ".pdf", bbox_inches='tight')
    plt.show()

    # Printing
    print("Importances")
    print("Mean\t\t\tstd")
    for k in sorted(imps_dict, key=imps_dict.get, reverse=True):
        print(k)
        print(str(imps_dict[k]) + ",\t" + str(imps_std_dict[k]))
        #print("\n")



# Plotting new
    keys_ordered = []
    keys_ordered_pretty = []
    for k in sorted(imps_dict, key=imps_dict.get, reverse=True):
        keys_ordered.append(k)
        keys_ordered_pretty.append(pretty_names_parameters[k])

    fig, ax = plt.subplots(figsize=(8, 8))
    bar_x = np.arange(len(keys_ordered))
    bar_y = [imps_dict[k] for k in keys_ordered]
    bar_err = [imps_std_dict[k] for k in keys_ordered]
    ax.barh(bar_x, bar_y, xerr=bar_err)
    ax.set_yticks(bar_x)
    ax.set_yticklabels(keys_ordered_pretty)
    ax.set_title("Parameter importance in PROSAIL simulations (" + pretty_names_func[str(loss_func)] + ")")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Parameter")

    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    for a in ax.get_xticklabels():
        a.set_fontsize(20)
    for a in ax.get_yticklabels():
        a.set_fontsize(20)


    if(save_fig):
        plt.savefig("/scratch/arp/domain/results/" + "parameter_importance_" + str(loss_func) + "_horizontal.pdf", bbox_inches='tight')
    plt.show()