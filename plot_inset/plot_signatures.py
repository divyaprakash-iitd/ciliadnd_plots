#!/usr/bin/env python3
import pandas as pd
import h5py
import numpy as np
import pickle
from generate_signatures import find_extreme_window
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from h5_gen_trajectory import create_visualization
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams['font.size'] = 24

# Constants
H = 0.02
L = H/2.75

# Read the base values
base_values = np.loadtxt('base_values.txt', delimiter=',')

with open('simdata_tip.pkl', 'rb') as f:
    simdata_tip = pickle.load(f)
    ciliano = 2
    for icilia in [ciliano]:
        # print(icilia)
        nsim = len(simdata_tip)
        
        # Specify the component 
        xidx, yidx = 0, 1
        compidx = yidx
        compmaskidx = (compidx % 2) + 2
        
    
        # Measure from the datum
        datum = base_values[icilia,compidx]

        # compmaskidx = ymaskidx
        simno = 15
        for isim in [simno]:
            # Convert the mask to bool
            boolmask = simdata_tip[isim][:,icilia,compmaskidx].astype(bool)
            
            # Extract the signal
            xorg = simdata_tip[isim][:,icilia,compidx]

xorg = xorg/L            
datum = datum/L

# Plot the datum
plt.axhline(datum,color='k',linestyle='-.',label='Datum')

# Plot the original signal
plt.plot(xorg,color='b',label='Signal',linewidth=2)

x = xorg.copy()

for i, bval in enumerate(boolmask):
    if bval:
        x[i] = xorg[i]
    else:
        x[i] = np.nan
# Plot the signature
scolor = 'g'
plt.plot(x,'-o',color=scolor,label='Signature')

# Plot window boundaries
true_idx = np.where(boolmask)[0]
left_idx = true_idx[0]
right_idx = true_idx[-1]
plt.axvline(x=left_idx, color='k', linestyle=':', alpha=0.7, label='Window boundaries')
plt.axvline(x=right_idx, color='k', linestyle=':', alpha=0.7)
 
# Plot the extremum
extremum_idx_array = [np.argmax(xorg), np.argmin(xorg)]
extremum_idx = extremum_idx_array[compidx%2]
extremum_value = xorg[extremum_idx] 
#plt.scatter(extremum_idx, extremum_value, color='m', s=100, marker='X', label=f"extremum")
plt.xlabel('Time Index')
plt.ylabel('y/L')

plt.gcf().set_size_inches(24,10)
fig_main, ax_main = plt.gcf(), plt.gca()
plt.legend(ncol=2)
plt.savefig('signature.pdf',dpi=300)

## Inset
path = f'sim_00{simno}/sim_00/database.h5'
limits = [left_idx, right_idx]
#
## Create an inset Axes inside the main Axes
##axins = inset_axes(ax_main, width="0.01%", height="0.01%", loc='lower right')
#axins = inset_axes(
#            ax_main, 
#            width=1.0, 
#            height=0.8, 
#            loc='lower right',
#            bbox_to_anchor=(0.65,0.05,0.3,0.3),
#            bbox_transform=ax_main.transAxes,
#            borderpad=0
#)
#
create_visualization(path,scolor,ciliano,limits)
plt.show()
