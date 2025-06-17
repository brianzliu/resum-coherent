import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import sys
import os
from . import plotting_utils as plotting
import pandas as pd
import matplotlib.lines as mlines

def plot(prediction_y_training, target_y_training, loss_training, prediction_y_testing, target_y_testing,  loss_testing, target_range=[0,1], it=None):

        index_list_1 = np.where(target_y_testing > 0.5)[0] if target_range[1] == 1 else []
        target_signal_testing = np.array([target_y_testing[i] for i in index_list_1])
        prediction_signal_testing = np.array([prediction_y_testing[i] for i in index_list_1])
        target_bkg_testing = np.delete(target_y_testing, index_list_1)
        prediction_bkg_testing = np.delete(prediction_y_testing, index_list_1)

        #######################

        index_list_2 = np.where(target_y_training > 0.5)[0] if target_range[1] == 1 else []
        target_signal_training = np.array([target_y_training[i] for i in index_list_2])
        prediction_signal_training = np.array([prediction_y_training[i] for i in index_list_2])
        target_bkg_training = np.delete(target_y_training, index_list_2)
        prediction_bkg_training = np.delete(prediction_y_training, index_list_2)  

        fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
        if it != None:
                fig.suptitle(f'Training Iteration {it}', fontsize=10)
        # plot testing 
        bins = int(100)
        if len(target_signal_testing)>0:
                ax[1].hist(target_signal_testing, range=target_range,bins=bins, color='orangered', alpha=1.0, label='label (signal)')
        ax[1].hist(target_bkg_testing, range=target_range,bins=bins, color=(3/255,37/255,46/255), alpha=0.8, label='label (bkg)')
        ax[1].hist(prediction_bkg_testing, range=target_range,bins=bins, color=(113/255,150/255,159/255), alpha=0.8, label='network (bkg)')
        if len(target_signal_training)>0:
                ax[1].hist(prediction_signal_testing, range=target_range,bins=bins,  color='coral', alpha=0.8,label='network (signal)')    
       

        ax[1].set_yscale('log')
        ax[1].set_ylabel("Count")
        ax[1].set_xlabel(r'$y_{CNP}$')
        ax[1].set_title(f'Testing (loss {loss_testing})', fontsize=10)
        #ax[1].legend(loc='bottom left', bbox_to_anchor=(0, -1.5))
        #ax[1].text(0.35,-0.25, f'Test data {Counter([i[0] for i in target_y_testing])}', fontsize=8.5, transform=ax[1].transAxes)

                      

        #plot training
        bins = int(100)

        if len(target_signal_training)>0:
                ax[0].hist(target_signal_training, range=target_range,bins=bins, color='orangered', alpha=1.0, label='label (signal)')
        ax[0].hist(target_bkg_training, range=target_range,bins=bins, color=(3/255,37/255,46/255), alpha=0.8, label='label (bkg)')
        ax[0].hist(prediction_bkg_training, range=target_range,bins=bins, color=(113/255,150/255,159/255), alpha=0.8, label='network (bkg)')
        if len(target_signal_training)>0:
                ax[0].hist(prediction_signal_training, range=target_range,bins=bins, color='coral', alpha=0.8, label='network (signal)')
        
        ax[0].set_yscale('log')
        ax[0].set_ylabel("Count")
        ax[0].set_xlabel(r'$y_{CNP}$')
        ax[0].set_title(f'Training (loss {loss_training})', fontsize=10)

        fig.subplots_adjust(bottom=0.3, wspace=0.33)
        if len(target_signal_training)>0:
                ax[1].legend(labels=['label (signal)', 'label (bkg)', 'network (bkg)', 'network (signal)'],loc='upper center', 
                bbox_to_anchor=(-0.15, -0.1),fancybox=False, shadow=False, ncol=2)
        else:
                ax[1].legend(labels=['label', 'network'],loc='upper center', 
                bbox_to_anchor=(-0.15, -0.1),fancybox=False, shadow=False, ncol=2)
        plt.show()

        return fig


def plot_config(prediction_y, target_y, loss, param=""):
    index_list = np.where(target_y == 1)[0]
    target_signal = np.array([target_y[i] for i in index_list])
    prediction_signal = np.array([prediction_y[i] for i in index_list])
    target_bkg = np.delete(target_y, index_list)
    prediction_bkg = np.delete(prediction_y, index_list)
    
    #fig, ax = plt.subplots(nrows=2, figsize=(6,6))
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4,1]})
    
    ax[0].hist(target_bkg, range=[0.0, 1.0],bins=100, color=(3/255,37/255,46/255), alpha=0.8, label='label (bkg)')
    ax[0].hist(prediction_bkg, range=[0.0, 1.0],bins=100, color=(113/255,150/255,159/255), alpha=0.8, label='network (bkg)')
    ax[0].hist(target_signal, range=[0.0, 1.0],bins=100, color='orangered', alpha=1.0, label='label (signal)')
    ax[0].hist(prediction_signal, range=[0.0, 1.0],bins=100, color='coral', alpha=0.8, label='network (signal)')
    
    
    ax[0].set_title(f'Testing (loss {loss}) {param}{Counter([i[0] for i in target_y])}', fontsize=10)
    ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Count")
    ax[0].set_xlabel(r"$y_{CNP}$")
    ax[1].set_axis_off()

    return fig



def sum_hist(prediction_y, target_y, htsig, htbkg, hpsig, hpbkg):

    index_list = np.where(target_y > 0.5)[0]
    target_signal = np.array([target_y[i] for i in index_list])
    prediction_signal = np.array([prediction_y[i] for i in index_list])
    target_bkg = np.delete(target_y, index_list)
    prediction_bkg = np.delete(prediction_y, index_list)

    bins = len(htsig)
    range = [0.0,1.0]

    h = np.histogram(target_signal, range=range,bins=bins)[0]
    htsig = htsig + h
    h = np.histogram(target_bkg, range=range, bins=bins)[0]
    htbkg = htbkg + h
    h = np.histogram(prediction_signal, range=range, bins=bins)[0]
    hpsig = htsig + h
    h += np.histogram(prediction_bkg, range=range, bins=bins)[0]
    hpbkg = hpbkg + h

    return htsig, htbkg, hpsig, hpbkg


def plot_result_summed(htsig, htbkg, hpsig, hpbkg):
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4,1]})
    nbins = len(htsig)
    range = [0.0, 1.0]
    bin_length = (range[1]-range[0])/nbins
    bins = np.arange(range[0],range[1]+bin_length, bin_length)
    centroids = (bins[1:] + bins[:-1]) / 2

    ax[0].hist(centroids, weights = htbkg, range=range, bins=nbins, color=(3/255,37/255,46/255), alpha=0.8, label='label (bkg)')
    ax[0].hist(centroids, weights = hpbkg, range=range, bins=nbins, color=(113/255,150/255,159/255), alpha=0.8, label='network (bkg)')
    ax[0].hist(centroids, weights = htsig, range=range, bins=nbins, color='orangered', alpha=1.0, label='label (signal)')
    ax[0].hist(centroids, weights = hpsig, range=range, bins=nbins, color='coral', alpha=0.8, label='network (signal)')
    
    ax[0].set_title(f'Testing', fontsize=10)
    ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Count")
    ax[0].set_xlabel(r"$y_{CNP}$")


    ax[1].set_axis_off()

    return fig

def get_subplot_result_configwise(ax, prediction_y, target_y, loss, param=""):
    index_list = np.where(target_y == 1)[0]

    target_signal = np.array([target_y[i] for i in index_list])
    prediction_signal = np.array([prediction_y[i] for i in index_list])
    target_bkg = np.delete(target_y, index_list)
    prediction_bkg = np.delete(prediction_y, index_list)
    
    ax.hist(target_bkg, range=[0.0, 1.0],bins=100, color=(3/255,37/255,46/255), alpha=0.8, label='label (bkg)')
    ax.hist(prediction_bkg, range=[0.0, 1.0],bins=100, color=(113/255,150/255,159/255), alpha=0.8, label='network (bkg)')
    ax.hist(target_signal, range=[0.0, 1.0],bins=100, color='orangered', alpha=1.0, label='label (signal)')
    ax.hist(prediction_signal, range=[0.0, 1.0],bins=100, color='coral', alpha=0.8, label='network (signal)')
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)
    ax.set_yscale("log")
    ax.set_ylabel("Count")
    ax.set_xlabel(r"$y_{CNP}$")



def plot_result_configwise(prediction_y, target_y, loss, x):
    sum_sim=np.sum(target_y)
    mean_pred=np.mean(prediction_y)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    fig.suptitle(f'Testing (loss {loss}) {Counter([i[0] for i in target_y])} mean(CNP): {mean_pred:.3f}, sum(Sim): {sum_sim:.0f}', fontsize=12)
    get_subplot_result_configwise(ax[0],prediction_y, target_y, loss)

    plotting.get_subplot_moderator(ax[1],x)
    tmp_str=f"r={x[0]}, d={x[1]}, n={x[2]}, " + r"$\varphi$=" + f"{x[3]}, L={x[4]}"
    ax[1].legend([tmp_str],loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)
        
    return fig

def get_marginalized(x_data, y_data, x_min, x_max, keep_axis, grid_steps=100):
        """
        Marginalizes predictions over all but one feature using random sampling when only one 
        prediction is available per sample (y_hf has shape (n_samples, 1)).
        """

        x_keep = x_data[:, keep_axis]

        # Define bins along the kept axis in the original scale.
        bin_edges = np.linspace(x_min, x_max, grid_steps + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # For each bin, compute the median and 1σ percentiles (16th and 84th) from the samples in the bin.
        medians = np.empty(grid_steps)
        lower_vals = np.empty(grid_steps)
        upper_vals = np.empty(grid_steps)
        for i in range(grid_steps):
            # Use a half-open interval except for the last bin.
            if i < grid_steps - 1:
                mask = (x_keep >= bin_edges[i]) & (x_keep < bin_edges[i+1])
            else:
                mask = (x_keep >= bin_edges[i]) & (x_keep <= bin_edges[i+1])
            if np.sum(mask) > 0:
                bin_values = y_data[mask]
                medians[i] = np.median(bin_values)
                lower_vals[i] = np.percentile(bin_values, 16)
                upper_vals[i] = np.percentile(bin_values, 84)
            else:
                medians[i] = np.nan
                lower_vals[i] = np.nan
                upper_vals[i] = np.nan

        # Compute errors for plotting (errorbars represent the distance from the median to the percentiles)
        lower_error = medians - lower_vals
        upper_error = upper_vals - medians
        
        return bin_centers, medians, lower_error, upper_error

def get_marginialized_all(config_file, grid_steps=100):
        x_labels = config_file["simulation_settings"]["theta_headers"]
        x_min = config_file["simulation_settings"]["theta_min"]
        x_max = config_file["simulation_settings"]["theta_max"]
        path_out  = config_file["path_settings"]["path_out_cnp"]
        version   = config_file["path_settings"]["version"]
        data=pd.read_csv(f'{path_out}/cnp_{version}_output.csv')
        y_label_sim = 'y_raw'
        y_label_cnp = 'y_cnp'
        fidelities  = np.unique(config_file["path_settings"]["fidelity"])  # this expects data to have both fidelities
        x_data = []
        y_data = []
        y_sim = []
        y_data_min=1.
        y_data_max=0.
        y_sim_min=1.
        y_sim_max=0.
        for f in fidelities:
                x_data.append(data.loc[(data['fidelity']==f)][x_labels].to_numpy())
                y_data.append(data.loc[(data['fidelity']==f)][y_label_cnp].to_numpy())
                y_sim.append(data.loc[(data['fidelity']==f)][y_label_sim].to_numpy())
                y_data_min=min(y_data_min,np.min(y_data[-1]))
                y_data_max=max(y_data_max,np.max(y_data[-1]))
                y_sim_min=min(y_sim_min,np.min(y_sim[-1]))
                y_sim_max=max(y_sim_max,np.max(y_sim[-1]))

        colors=[["salmon","darkturquoise"],["darkred","teal"],["orangedred","darkslategrey"]]
        markers=['.','s','o''x']
        grid_steps = 100
        n_params = len(x_labels)
        n_cols = 2
        n_rows = int(np.ceil(n_params / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

        for i, param in enumerate(x_labels):
                for f in range(len(x_data)):
                        row = i // n_cols
                        col = i % n_cols
                        ax1 = axes[row][col]

                        # Get marginalized draws
                        x_cnp, y_cnp,_,_ = get_marginalized(x_data=x_data[f], y_data=y_data[f], x_min=x_min[i], x_max=x_max[i], keep_axis=i, grid_steps=grid_steps)
                        x_raw, y_raw,_,_ = get_marginalized(x_data=x_data[f], y_data=y_sim[f],  x_min=x_min[i], x_max=x_max[i], keep_axis=i, grid_steps=grid_steps)

                        # Plot on first axis
                        p1 = ax1.scatter(x_cnp, y_cnp, color=colors[f][0],alpha=1., marker=markers[f])
                        if (f==0):
                                ax1.set_ylabel('$y_{cnp}$', color=colors[f][0])
                                ax1.tick_params(axis='y', labelcolor=colors[f][0])
                                ax1.set_xlabel(param)
                                ax1.set_ylim(y_data_min*0.9,y_data_max*1.005)
                                ax2 = ax1.twinx()
                                ax2.set_ylabel('$y_{raw}$', color=colors[f][1])
                                ax2.set_ylim(y_sim_min*0.8,y_sim_max*1.005)
                                ax2.tick_params(axis='y', labelcolor=colors[f][1])

                        # Plot on second axis
                        
                        p2 = ax2.scatter(x_raw, y_raw, color=colors[f][1],alpha=1., marker=markers[f])


        handles = []
        for j in range(len(x_data)):
                handles.append(mlines.Line2D([], [], color=colors[j][0], marker=markers[j], linestyle='None', label='$y_{cnp}$'+f'(f={j})'))
                handles.append( mlines.Line2D([], [], color=colors[j][1], marker=markers[j], linestyle='None', label='$y_{raw}$'+f'(f={j})'))
        fig.legend(
                handles=handles,
                loc='lower center',
                ncol=2*len(x_data),
                title='',
                bbox_to_anchor=(0.5, -0.05)
                )

        # Adjust layout
        plt.tight_layout()
        return fig