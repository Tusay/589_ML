import numpy as np
import glob
import os
from blimpy import Waterfall
import blimpy as bl
import gc
import time
import matplotlib
import matplotlib.pyplot as plt

start=time.time()

def get_elapsed_time(start=0):
    end = time.time() - start
    time_label = 'seconds'    
    if end > 3600:
        end = end/3600
        time_label = 'hours'
    elif end > 60:
        end = end/60
        time_label = 'minutes'
    return end, time_label

def plot_waterfall(wf, source_name=None, f_start=None, f_stop=None, **kwargs):
    # get plot data
    plot_f, plot_data = wf.grab_data(f_start=f_start, f_stop=f_stop)

    # determine extent of the plotting panel for imshow
    extent=(plot_f[0], plot_f[-1], (wf.timestamps[-1]-wf.timestamps[0])*24.*60.*60, 0.0)

    # plot and scale intensity (log vs. linear)
    kwargs['cmap'] = kwargs.get('cmap', 'viridis')
    plot_data = 10.0 * np.log10(plot_data)

    # get normalization parameters
    vmin = plot_data.min()
    vmax = plot_data.max()
    normalized_plot_data = (plot_data - vmin) / (vmax - vmin)

    # display the waterfall plot
    this_plot = plt.imshow(normalized_plot_data,aspect='auto',rasterized=True,interpolation='nearest',extent=extent,**kwargs)

    del plot_f, plot_data

    return this_plot

def one_wf_plot(fil_file_list,f_start=0,f_stop=0,drift_rate=0,source_name_list=[],save_dir=None,save_name=None,extension='png'):
    # set up the sub-plots
    n_plots = len(fil_file_list)
    fig,ax = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))

    if not f_start or not f_stop:
        print('\nf_start and/or f_stop input error. Try again.\n')
        return None

    # define more plot parameters
    mid_f = np.abs(f_start+f_stop)/2.

    subplots = []

    # Fill in each subplot for the full plot
    for ii, filename in enumerate(fil_file_list):
        # identify panel
        subplot = plt.subplot(n_plots, 1, ii + 1)
        subplots.append(subplot)

        # read in data
        max_load = bl.calcload.calc_max_load(filename)
        wf = bl.Waterfall(filename, f_start=f_start, f_stop=f_stop, max_load=max_load)

        this_plot = plot_waterfall(wf, f_start=f_start, f_stop=f_stop)

        del wf
        gc.collect()

    plt.subplots_adjust(hspace=0,wspace=0)
    [axi.set_axis_off() for axi in ax.ravel()]
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    # save the figures
    if save_dir:
        if save_dir[-1] != "/":
            save_dir += "/"
        if save_name:
            path_png = save_dir+save_name
        else:
            path_png = save_dir + 'freq_' "{:0.6f}".format(mid_f) + '.' + extension
        plt.savefig(path_png, bbox_inches='tight',format=extension, pad_inches=0)

    return fig

blc32_f2_plots=sorted(glob.glob('/storage/home/nxt5197/work/PPO/TOI-216/node_by_node/blc32/Freqs_3904_to_4032/MJDate_59221/plots_TIC55652896_S_f2_snr10.0/*.png'))
os.chdir('/storage/home/nxt5197/scratch/PPO/TOI-216/')
fils=sorted(glob.glob('*h5'))
power_matrix = []
freq_ranges = []
counter_file = '/storage/home/nxt5197/work/589_Machine_Learning/plots/data/icounter.txt'
for i,png in enumerate(blc32_f2_plots):
    f_start = float(png.split('_freq_')[1].split('.png')[0])
    drift_rate = float(png.split('_freq_')[0].split('_dr_')[1])
    if abs(drift_rate) <= 0.10: 
        half_f = 250
    elif abs(drift_rate) == 0.11:
        half_f = 272
    elif abs(drift_rate) == 0.13:
        half_f = 304
    elif abs(drift_rate) == 0.14:
        half_f = 337
    elif abs(drift_rate) == 0.32:
        half_f = 769
    elif abs(drift_rate) == 9.14:
        half_f = 21969
    elif abs(drift_rate) == 24.31:
        half_f = 58424
    elif abs(drift_rate) == 36.21:
        half_f = 87026
    else:
        print('You fucked up. You got the drift rate and frequency range wrong. Try again, buddy.')
        break
    f_mid = f_start+half_f*1e-6
    delta_f = 500
    f_stop = f_mid+delta_f/2*1e-6
    f_start = f_mid-delta_f/2*1e-6
    save_dir='/storage/home/nxt5197/work/589_Machine_Learning/plots/imgs/'
    save_name=png.split('/')[-1]
    image = one_wf_plot(fils,f_start=f_start,f_stop=f_stop,save_dir=save_dir,save_name=save_name,extension='png')
    end, time_label = get_elapsed_time(start)
    with open(counter_file, 'w') as f:
        f.write(f'{i+1} of {len(blc32_f2_plots)} for loops completed in {end:.2f} {time_label}.')
os.chdir('/storage/home/nxt5197/work/589_Machine_Learning/plots/data/')
np.save('power_matrix.npy', power_matrix)
np.save('freq_ranges.npy', freq_ranges)