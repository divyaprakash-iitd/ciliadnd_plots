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
from mpl_toolkits.axes_grid1 import host_subplot

plt.rcParams['font.size'] = 24

# Constants
H = 0.02
L = H/2.75

# Read the base values
base_values = np.loadtxt('base_values.txt', delimiter=',')

def plot_signatures(ax1,simno=15,ciliano=2):
    with open('simdata_tip.pkl', 'rb') as f:
        simdata_tip = pickle.load(f)
        xidx, yidx = 0, 1
        # First y-axis
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('x/L')
        # Second y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('y/L')

        for compidx, ax, color in zip([xidx, yidx],[ax1, ax2], ['b', 'g']):
            for icilia in [ciliano]:
                # print(icilia)
                nsim = len(simdata_tip)
                
                # Specify the component 
                # xidx, yidx = 0, 1
                # compidx = yidx
                compmaskidx = (compidx % 2) + 2
                
                # Measure from the datum
                datum = base_values[icilia,compidx]

                # compmaskidx = ymaskidx
                for isim in [simno]:
                    # Convert the mask to bool
                    boolmask = simdata_tip[isim][:,icilia,compmaskidx].astype(bool)
                    
                    # Extract the signal
                    xorg = simdata_tip[isim][:,icilia,compidx]

            xorg = xorg/L            
            datum = datum/L

            # Plot the datum
            ax.axhline(datum,color='k',linestyle='-.',label='Datum')

            # Plot the masked signal
            p_h, = ax.plot(xorg[boolmask],color=color,label='Signal',linewidth=2)

            # Match curve and y label colors
            ax.yaxis.label.set_color(p_h.get_color())

        # Layout
        plt.tight_layout()

#fig, ax = plt.subplots()
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
#ax = host_subplot(111)

for icilia, iax in enumerate(ax.T.flatten()):
    plot_signatures(iax,simno=15,ciliano=icilia)

plt.show()
