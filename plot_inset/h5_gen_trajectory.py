import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
import h5py
from pathlib import Path
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import FancyArrowPatch

plt.rcParams['font.size'] = 18

def add_arrowheads(ax,x,y,n):
    npoints = len(x)
    offset = 10
    idvec = np.linspace(0+offset,npoints-offset,n,dtype=int)
    print(idvec)
    for i in range(n):
        start = (x[idvec[i]],y[idvec[i]])
        end = (x[idvec[i]+1],y[idvec[i]+1])
        # Create an arrow patch
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle='-|>',  # Arrow with a head
            mutation_scale=10,  # Size of the arrowhead
            color='b',
            linewidth=2
        )
        # Add the arrow to the plot
        ax.add_patch(arrow)


def fit_ellipse(x, y):
    """Fit an ellipse to a set of points."""
    x = x.flatten()
    y = y.flatten()
    
    # Calculate mean and center data
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    # Compute covariance matrix
    cov = np.cov(x_centered, y_centered)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate semi-major and semi-minor axes
    a = 2 * np.sqrt(eigenvalues[0])
    b = 2 * np.sqrt(eigenvalues[1])
    
    # Calculate angle
    theta = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    return x_mean, y_mean, a, b, np.degrees(theta)

def create_visualization(path, color, ciliano, limits=None, ax=None):
    """Create a single image showing particle and cilia at several time instances with trajectory."""
    # Define constants
    FIG_WIDTH, FIG_HEIGHT = 1920, 961
    DPI = 200
    figsize = (FIG_WIDTH/DPI, FIG_HEIGHT/DPI)
    PADDING = 0.05  # 5% padding for axis limits
    N_INSTANCES = 5  # Number of time instances to plot
    
    # Load mesh data
    Lx, Ly = 0.01, 0.002
    
    # Open the HDF5 file
    with h5py.File(path, "r") as f:
        # Get cilia data
        cilia_group = f["cilia"]
        cx_data = cilia_group["cx"][:]
        cy_data = cilia_group["cy"][:]
        lcilia = cilia_group.attrs["lcilia"]
        ncilia = cilia_group.attrs["ncilia"]
        
        # Get particle data
        particle_group = f["particles"]
        px_data = particle_group["px"][:]
        py_data = particle_group["py"][:]
        
        # Get time data
        time_group = f["time"]
        time_data = time_group["particles"][:]
        
    # Get total number of time steps
    nFiles = px_data.shape[0]
        
    # Use cilia length from HDF5 file
    L = lcilia
    
    # Calculate fixed axis limits with padding
    x_min = 0.0 / L - PADDING
    x_max = Lx / L + PADDING
    y_min = 0.0 / L - PADDING
    y_max = Ly / L + PADDING
    
    # Create figure
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=DPI)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    
    # Set axis labels
    plt.xlabel(r'$x/L$')
    plt.ylabel(r'$y/L$')
    
    # Add divider for consistent layout
    divider = make_axes_locatable(ax)
    
    # Add horizontal dividing line
    #ax.hlines(y=0.5*Ly/L, xmin=0.8*Lx/L, xmax=Lx/L, color='r', linewidth=1, linestyle='--')
    
    # Select uniformly spaced time indices
    if limits == None:
        t0, tend = 0, nFiles-1
    else:
        t0, tend = limits[0], limits[1]
    time_indices = np.linspace(t0, tend, N_INSTANCES, dtype=int)
    alphas = np.linspace(0.1, 0.7, N_INSTANCES)  # Transparency from most transparent to opaque
    
    # Compute particle trajectory (using first 30 points to compute centroid)
    trajectory_x = []
    trajectory_y = []
    for i in range(nFiles):
        centroid_x = np.mean(px_data[i, 0, :30]) / L
        centroid_y = np.mean(py_data[i, 0, :30]) / L
        trajectory_x.append(centroid_x)
        trajectory_y.append(centroid_y)
    
    # Plot trajectory
    ax.plot(trajectory_x, trajectory_y, 'b-.', linewidth=1.0, label='Particle Trajectory')
    
    # Add arrowheads
    add_arrowheads(ax,trajectory_x,trajectory_y,5)

    # Plot particle and cilia at selected instances
    for idx, alpha in zip(time_indices, alphas):
        print(f'Processing time index {idx}')
        
        # Plot cilia
        for j in range(ncilia):
            if j==ciliano:
                color = 'g'
            else:
                color = 'k'
            cilia_x = cx_data[idx, j, :] / L
            cilia_y = cy_data[idx, j, :] / L
            ax.plot(cilia_x, cilia_y, color=color, linewidth=2, alpha=alpha)
        
        # For particles, we'll use both polygon and fitted ellipse
        particle_x = px_data[idx, 0, :30] / L
        particle_y = py_data[idx, 0, :30] / L
        
        # Plot particle as polygon
        polygon = Polygon(np.column_stack((particle_x, particle_y)), 
                         facecolor='g', alpha=alpha)
        ax.add_patch(polygon)
        
        # Alternative: fit and plot ellipse
        # cx, cy, a, b, angle = fit_ellipse(particle_x, particle_y)
        # ellipse = Ellipse((cx, cy), a, b, angle=angle, 
        #                  facecolor='k', alpha=alpha)
        # ax.add_patch(ellipse)
        
    
    # Set aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Use consistent layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('simulation_snapshot.png', dpi=DPI, bbox_inches='tight')
    plt.savefig('simulation_snapshot.pdf', dpi=DPI, bbox_inches='tight')
    
    return plt.gcf(), plt.gca()    

if __name__ == "__main__":
    path = 'sim_0015/sim_00/database.h5'
    fig, ax = create_visualization(path,[10,100])
