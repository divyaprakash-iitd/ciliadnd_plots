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
import matplotlib.ticker as ticker

plt.rcParams['font.size'] = 18

# Constants
H = 0.02
L = H/2.75

# Read the base values
base_values = np.loadtxt('base_values.txt', delimiter=',')

def plot_signatures(ax1, simno=15, ciliano=2, show_xlabel=False, show_ylabel_left=False, show_ylabel_right=False):
    with open('simdata_tip.pkl', 'rb') as f:
        simdata_tip = pickle.load(f)
        xidx, yidx = 0, 1
        # First y-axis
        if show_xlabel:
            ax1.set_xlabel('Time Index')
        if show_ylabel_left:
            ax1.set_ylabel('x/L')
        # Second y-axis
        ax2 = ax1.twinx()
        if show_ylabel_right:
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

            x = xorg.copy()
            #############
            for i, bval in enumerate(boolmask):
                if bval:
                    x[i] = xorg[i]
                else:
                    x[i] = np.nan

            # Plot window boundaries
            true_idx = np.where(boolmask)[0]
            left_idx = true_idx[0]
            right_idx = true_idx[-1]
            #############

            # Plot the datum
            # ax.axhline(datum,color='k',linestyle='-.',label='Datum')

            # Plot the masked signal
            #p_h, = ax.plot(xorg[boolmask],color=color,label='Signal',linewidth=2)
            p_h, = ax.plot(x,color=color,label='Signal',linewidth=2)

            # Match curve and y label colors
            ax.yaxis.label.set_color(p_h.get_color())

            # Set decimal places on the y-ticks
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))  # 2 decimal places

        # Hide tick labels if not on edges
        if not show_xlabel:
            ax1.set_xticklabels([])
        if not show_ylabel_left:
            ax1.set_yticklabels([])
        if not show_ylabel_right:
            ax2.set_yticklabels([])

        # Layout
        plt.tight_layout()

##fig, ax = plt.subplots()
#fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
##ax = host_subplot(111)
#
#idcilia = [0,2,4,1,3,5]
#
#icilia = 0
#for irow in range(2):
#    for jcol in range(3):
#       
#        if irow == 1:
#            show_xlabel = True
#        else:
#            show_xlabel = False 
#        
#        if jcol == 0:
#            show_ylabel_left = True 
#        else:
#            show_ylabel_left = False 
#        
#        if jcol == 2:
#            show_ylabel_right = True 
#        else:
#            show_ylabel_right = False 
#        
#        plot_signatures(ax[irow,jcol], simno=15, ciliano=idcilia[icilia], 
#                       show_xlabel=show_xlabel, 
#                       show_ylabel_left=show_ylabel_left, 
#                       show_ylabel_right=show_ylabel_right)
#        icilia = icilia + 1

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
#idcilia = [0,2,4,1,3,5]
idcilia = [1,3,5,0,2,4]

for i, axis in enumerate(ax.flat):
    irow, jcol = divmod(i, 3)
    plot_signatures(axis, simno=15, ciliano=idcilia[i], 
                   show_xlabel=(irow == 1), 
                   show_ylabel_left=(jcol == 0), 
                   show_ylabel_right=(jcol == 2))
plt.savefig('signatures_grid.pdf',dpi=300)
plt.show()
