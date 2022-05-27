import numpy as np
import pandas as pd
import glob
import os
from blimpy import Waterfall
import blimpy as bl
import gc
import time

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

def get_data_matrix(fil_file_list,f_start=0,f_stop=0):
    freq_range=[0 for i in range(len(fil_file_list))]
    power=[[0,0] for i in range(len(fil_file_list))]
    if not f_start:
        wf=Waterfall(fil_file_list[0],load_data=False)
        f_start=wf.container.f_start
        del wf
        gc.collect()
    if not f_stop:
        wf=Waterfall(fil_file_list[0],load_data=False)
        f_stop=wf.container.f_stop
        del wf
        gc.collect()
    for i,filename in enumerate(fil_file_list):
        max_load = bl.calcload.calc_max_load(filename)
        wf = bl.Waterfall(filename, f_start=f_start, f_stop=f_stop, max_load=max_load)
        freq_range[i] = wf.grab_data()[0]
        power[i] = wf.grab_data()[1]
    power = np.array(power)
    power = np.reshape(power,(np.shape(power)[0]*np.shape(power)[1],np.shape(power)[2]))
    return power,freq_range[0]

blc32_f2_plots=sorted(glob.glob('/storage/home/nxt5197/work/PPO/TOI-216/node_by_node/blc32/Freqs_3904_to_4032/MJDate_59221/plots_TIC55652896_S_f2_snr10.0/*.png'))
os.chdir('/storage/home/nxt5197/scratch/PPO/TOI-216/')
fils=sorted(glob.glob('*h5'))
power_matrix = []
freq_ranges = []
counter_file = '/storage/home/nxt5197/work/589_Machine_Learning/plots/data/counter.txt'
for i,png in enumerate(blc32_f2_plots):
    f_start = float(png.split('_freq_')[1].split('.png')[0])
    drift_rate = float(png.split('_freq_')[0].split('_dr_')[1])
    if abs(drift_rate) <= 0.10: 
        half_f = 250
    elif abs(drift_rate) == 0.11:
        half_f = 272
    elif abs(drift_rate) == 0.12:
        half_f = 288
    elif abs(drift_rate) == 0.13:
        half_f = 304
    elif abs(drift_rate) == 0.14:
        half_f = 337
    elif abs(drift_rate) == 0.15:
        half_f = 369
    elif abs(drift_rate) == 0.16:
        half_f = 385
    elif abs(drift_rate) == 0.17:
        half_f = 417
    else:
        print(f'\nDrift rate too high: {drift_rate}\nSkipping.\n')
        continue
    # elif abs(drift_rate) == 0.32:
    #     half_f = 769
    # elif abs(drift_rate) == 9.14:
    #     half_f = 21969
    # elif abs(drift_rate) == 24.31:
    #     half_f = 58424
    # elif abs(drift_rate) == 36.21:
    #     half_f = 87026
    # else:
    #     print('You fucked up. You got the drift rate and frequency range wrong. Try again, buddy.')
    #     break
    f_mid = f_start+half_f*1e-6
    delta_f = 500
    f_stop = f_mid+delta_f/2*1e-6
    f_start = f_mid-delta_f/2*1e-6
    # print(f'f_start: {f_start}\tf_stop: {f_stop}')
    powers,freqs = get_data_matrix(fils,f_start=f_start,f_stop=f_stop)
    pnorm = (powers - powers.min())/(powers.max()-powers.min())
    power_matrix.append(pnorm)
    freq_ranges.append(freqs)
    end, time_label = get_elapsed_time(start)
    with open(counter_file, 'w') as f:
        f.write(f'{i+1} of {len(blc32_f2_plots)} for loops completed in {end:.2f} {time_label}.')
os.chdir('/storage/home/nxt5197/work/589_Machine_Learning/plots/data/')
np.save('power_matrix.npy', power_matrix)
np.save('freq_ranges.npy', freq_ranges)
pd.DataFrame(power_matrix).to_csv("power_matrix.csv")
pd.DataFrame(freq_ranges).to_csv("freq_ranges.csv")