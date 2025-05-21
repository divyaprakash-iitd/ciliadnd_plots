<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Project README

This repository contains scripts for visualizing and analyzing simulation data related to particle and cilia dynamics, as well as flow fields. Below is a brief outline of what each script does.

---

## **Scripts Overview**

### **1. h5_gen_trajectory.py**

- **Purpose:** Generates a visualization of particle and cilia motion over time using data from an HDF5 file (`database.h5`).
- **Key Features:**
    - Plots the trajectory of a particle as it moves, adding arrowheads to indicate direction.
    - Overlays snapshots of the particle and cilia at multiple time instances.
    - Fits and optionally plots an ellipse to the particle shape at selected times.
    - Saves the visualization as both PNG and PDF images (`simulation_snapshot.png`, `simulation_snapshot.pdf`)[^1].

---

### **2. plot_image.py**

- **Purpose:** Visualizes flow simulations by plotting streamlines of the velocity field, with optional video creation.
- **Key Features:**
    - Loads mesh and simulation data from text files.
    - Interpolates velocity data to a uniform grid for smooth streamline plotting.
    - Plots cilia positions and overlays streamlines at each time step.
    - Can save a single frame (as PNG/PDF) for a specified time step or generate a video of the entire simulation (`simulation_with_streamlines.mp4`).
    - Command-line interface allows selection of a specific time step for visualization[^2].

---

### **3. plot_matlab.m**

- **Purpose:** MATLAB script for plotting velocity streamlines and cilia motion from simulation outputs.
- **Key Features:**
    - Reads mesh and velocity data, computes cell-centered velocities.
    - Plots streamlines of the flow field at each time step.
    - Overlays cilia positions and finite element mesh.
    - Creates a video of the streamline evolution (`ibm_c_streamlines.avi`)[^3].

---

## **Summary Table**

| Script Name | Language | Main Functionality | Output |
| :-- | :-- | :-- | :-- |
| h5_gen_trajectory.py | Python | Particle/cilia trajectory visualization from HDF5 | PNG/PDF images |
| plot_image.py | Python | Streamline visualization and video from simulation data | PNG/PDF images, MP4 video |
| plot_matlab.m | MATLAB | Streamline/cilia visualization and video | AVI video |


---

## **Usage**

- Ensure all required data files (e.g., mesh files, HDF5 database, simulation outputs) are present in the working directory.
- For Python scripts, install necessary dependencies (numpy, matplotlib, h5py, cv2, scipy, etc.).
- For `plot_image.py`, use the command line to specify options:

```bash
python plot_image.py --time_step 10
```

- For MATLAB, run `plot_matlab.m` directly in the MATLAB environment.

---

These scripts provide a comprehensive toolkit for visualizing and analyzing particle, cilia, and flow field dynamics in simulation studies.

<div style="text-align: center">⁂</div>

[^1]: h5_gen_trajectory.py

[^2]: plot_image.py

[^3]: plot_matlab.m

