import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2
from pathlib import Path
import glob
from io import BytesIO
import f90nml
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse

plt.rcParams['font.size'] = 18

def cell_center(u, v):
    """Convert staggered velocity to cell-centered velocity."""
    Ny, Nx = u.shape[0] - 2, u.shape[1] - 1
    uc = u[1:Ny+1, :Nx]
    vc = v[:Ny, 1:Nx+1]
    return uc, vc

def interpolate_to_uniform_grid(x, y, values, grid_size=50):
    """Interpolate data to a uniform grid for streamplot."""
    xi = np.linspace(np.min(x), np.max(x), grid_size)
    yi = np.linspace(np.min(y), np.max(y), grid_size)
    
    # Create meshgrid for interpolation
    X, Y = np.meshgrid(xi, yi)
    
    # Use simple method for interpolation - replace with scipy if available
    from scipy.interpolate import griddata
    grid_values = griddata((x.flatten(), y.flatten()), values.flatten(), (X, Y), method='linear')
    
    return xi, yi, grid_values

def load_mesh_data():
    """Load mesh data from text files."""
    xu = np.loadtxt('u_x_mesh.txt')
    yu = np.loadtxt('u_y_mesh.txt')
    xv = np.loadtxt('v_x_mesh.txt')
    yv = np.loadtxt('v_y_mesh.txt')
    xp = np.loadtxt('p_x_mesh.txt')
    yp = np.loadtxt('p_y_mesh.txt')
    return xu, yu, xv, yv, xp, yp

def setup_video_writer(fig_width, fig_height):
    """Set up video writer for saving the simulation."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter('simulation_with_streamlines.mp4', fourcc, 24, (fig_width, fig_height), isColor=True)

def create_visualization(time_step=None):
    """
    Create visualization of the simulation with streamlines.
    
    Parameters:
    -----------
    time_step : int or None
        Specific time-step to visualize. If None, creates video of all time steps.
    """
    # Set consistent style
    #plt.style.use('default')
    
    # Define constants
    FIG_WIDTH, FIG_HEIGHT = 1920, 961
    DPI = 200
    figsize = (FIG_WIDTH/DPI, FIG_HEIGHT/DPI)
    PADDING = 0.05  # 5% padding for axis limits
    
    # Load mesh data
    xu, yu, xv, yv, xp, yp = load_mesh_data()
    Lx, Ly = np.max(xu), np.max(yv)
    L = Ly / 2.75
    
    # Get file lists
    u_files = sorted(glob.glob('u_0*'))
    v_files = sorted(glob.glob('v_0*'))
    c_files = sorted(glob.glob('C_0*'))
    p_files = sorted(glob.glob('P_0*'))
    nFiles = len(u_files)
    
    # Read initial data for dimensions
    u = np.loadtxt(u_files[0])
    Nx, Ny = u.shape[1] - 1, u.shape[0] - 2
    
    # Load additional data
    M = np.loadtxt("MP.txt")
    iskip = 1
    
    # Read simulation parameters
    nml = f90nml.read("input_params.dat")
    nu = nml['flow']['nu']
    dt = nml['time']['dt']
    it_save = nml['time']['it_save']
    
    # Calculate fixed axis limits with padding
    x_min = np.min(xu) / L - PADDING
    x_max = np.max(xu) / L + PADDING
    y_min = np.min(yu) / L - PADDING
    y_max = np.max(yu) / L + PADDING
    
    # Precompute contour levels
    u_min, u_max = np.inf, -np.inf
    for iFile in range(0, nFiles, iskip):
        u = np.loadtxt(u_files[iFile])
        u_min = min(u_min, np.min(u * L / nu))
        u_max = max(u_max, np.max(u * L / nu))
    levels = np.linspace(u_min, u_max, 31)  # 30 contours needs 31 levels
    
    # Create a centered figure 
    def create_figure_for_timestep(iFile):
        # Load and process velocity data
        u = np.loadtxt(u_files[iFile])
        v = np.loadtxt(v_files[iFile])
        uc, vc = cell_center(u, v)
        
        # Create figure with fixed size
        fig = plt.figure(figsize=figsize, dpi=DPI)
        ax = fig.add_subplot(111)
        
        # Create contour plot with fixed levels
        # contour = ax.contourf(xu/L, yu/L, u*L/nu, levels=levels, cmap='viridis', alpha=0.7)
        
        # Calculate grid for streamlines
        # For streamlines, we need to interpolate the velocity field to a regular grid
        XU, YU = np.meshgrid(xu[0, :]/L, yu[:, 0]/L)
        XV, YV = np.meshgrid(xv[0, :]/L, yv[:, 0]/L)
        
        # Interpolate velocities to common grid
        x_stream = np.linspace(x_min+PADDING, x_max-PADDING, 100)
        y_stream = np.linspace(y_min+PADDING, y_max-PADDING, 100)
        X_stream, Y_stream = np.meshgrid(x_stream, y_stream)
        
        # Use scipy's griddata for interpolation
        from scipy.interpolate import griddata
        U_stream = griddata((XU.flatten(), YU.flatten()), 
                            u.flatten(), 
                            (X_stream, Y_stream), 
                            method='linear')
        V_stream = griddata((XV.flatten(), YV.flatten()), 
                            v.flatten(), 
                            (X_stream, Y_stream), 
                            method='linear')
        
        # Plot streamlines
        density = 3
        lw = 0.5
        arrowsize = 0.5
        streamplot = ax.streamplot(x_stream, y_stream, U_stream, V_stream, 
                                  density=density, color='blue', linewidth=lw, 
                                  arrowsize=arrowsize)
        
        # Set axis labels
        plt.xlabel(r'$x/L$')
        plt.ylabel(r'$y/L$')
        
        ## Add colorbar with fixed formatting
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="3%", pad=0.1)
        #cbar = plt.colorbar(contour, cax=cax, format="%02.1f")
        #cbar.set_label(r'$uL/\nu$', rotation=90, labelpad=5)
        
        ## Plot mesh
        #ax.plot(xp/L, yp/L, 'w-', alpha=0.0)
        #
        ## Plot FEM mesh
        #P = np.loadtxt(p_files[iFile])
        #polygon = Polygon(np.column_stack((P[0, :30]/L, P[1, :30]/L)), 
        #                facecolor='#edb120', alpha=1)
        #ax.add_patch(polygon)
        #
        ## Plot dividing lines with fixed positions
        #ax.plot([Lx, Lx]/L, [0*Ly, 0.5*Ly]/L, 'r-', linewidth=2)
        #ax.plot([Lx, Lx]/L, [0.5*Ly, 1.0*Ly]/L, 'k-', linewidth=2)
        
        # Plot cilia
        C = np.loadtxt(c_files[iFile]).T
        for i in range(0, C.shape[1], 2):
            #ax.plot(C[:, i]/L, C[:, i+1]/L, 'b-', linewidth=2)
            ax.scatter(C[:, i]/L, C[:, i+1]/L, s=5,  c='k')
        
        # Set fixed title format
        time_value = iFile * dt * it_save
        print(f"Time: {time_value} seconds")
        # ax.set_title(f'Time: {time_value:.3f} s (Frame {iFile})', pad=5)
        
        # Set fixed aspect ratio and limits
        ax.set_aspect('equal')
        # ax.set_xlim(x_min, x_max)
        ax.set_xlim(6.0, 9.0)
        ax.set_ylim(y_min, y_max)
        
        # Use consistent layout
        plt.tight_layout()
        
        return fig, time_value
    
    # If specific time step is requested, only plot that one
    if time_step is not None:
        if time_step < 0 or time_step >= nFiles:
            print(f"Error: Requested time step {time_step} is out of range (0-{nFiles-1})")
            return
        
        fig, time_value = create_figure_for_timestep(time_step)
        output_file = f'streamlines_time_{time_value:.3f}.png'
        output_file_pdf = f'streamlines_time_{time_value:.3f}.pdf'
        fig.savefig(output_file, dpi=DPI, bbox_inches='tight')
        fig.savefig(output_file_pdf, bbox_inches='tight')
        print(f"Saved streamline visualization for time {time_value:.3f}s to {output_file}")
        plt.close(fig)
        return
    
    # Otherwise create video with all frames
    try:
        # Set up video writer
        out = setup_video_writer(FIG_WIDTH, FIG_HEIGHT)
        
        for iFile in range(0, nFiles, iskip):
            print(f'Processing file # {iFile} of {nFiles}')
            
            fig, _ = create_figure_for_timestep(iFile)
            
            # Convert to image and write to video
            try:
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Direct buffer conversion failed, trying alternative method: {e}")
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=DPI)
                buf.seek(0)
                img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (FIG_WIDTH, FIG_HEIGHT))
                buf.close()
            
            # Write frame to video
            out.write(image)
            plt.close(fig)
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        raise
        
    finally:
        if 'out' in locals():
            out.release()
        plt.close('all')
        print("Visualization with streamlines completed.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize flow with streamlines')
    parser.add_argument('--time_step', type=int, default=None, 
                        help='Specific time step to visualize (0-based index)')
    args = parser.parse_args()
    
    create_visualization(time_step=args.time_step)
    #create_visualization(time_step=20)
