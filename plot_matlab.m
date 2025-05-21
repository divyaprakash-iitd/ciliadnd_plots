clear; clc; close all;

% Description: Plots the velocity profiles (now streamlines) for the fundamental solution of
% stoke's flow

nml = read_namelist("input_params.dat");

% Load the data
xu = load('u_x_mesh.txt');
yu = load('u_y_mesh.txt');
xv = load('v_x_mesh.txt');
yv = load('v_y_mesh.txt');
xp = load('p_x_mesh.txt'); % Mesh for pressure, assumed to be cell centers for velocity
yp = load('p_y_mesh.txt'); % Mesh for pressure, assumed to be cell centers for velocity

% Domain size
Lx = max(max(xu));
Ly = max(max(yv));

uFile = dir(strcat('u_0','*'));
vFile = dir(strcat('v_0','*'));
CFile = dir(strcat('C_0','*'));
PFile = dir(strcat('P_0','*'));

% Load one file to get dimensions (Nx, Ny)
temp_u = load(uFile(1).name); % Load the first u-file to determine sizes
Nx = size(temp_u,2)-1;
Ny = size(temp_u,1)-2;

% Initialize cell-centered velocity arrays
uc = zeros(Ny,Nx);
vc = zeros(Ny,Nx);

% Total number of files
nFiles = length(uFile);

% Visualize cilia motion over velocity field (now streamlines)
vid = VideoWriter('ibm_c_streamlines.avi','Uncompressed AVI'); % Changed video name
vid.FrameRate = 24;
open(vid);
figure(1)
fig = gcf;
fig.Position = [1 1 1920 961]; % Existing figure position

% Commented out cmin, cmax calculation as it was for contourf and partially commented itself
% [cmin, cmax] = deal(0);
% xmin=max(max(xu)); % These seem to be legacy or for specific debugging
% xmax=min(min(xu));

% xmin=Lx/2-0.001*Lx; % These were for a specific xlim, also commented out later
% xmax=Lx/2+0.001*Lx;

tdef = zeros(1,nFiles);
iskip=max(round(nFiles*1.0/20.0),1);
iskip = 10; % This determines which frames are processed

for iFile = 9 %1:iskip:nFiles
    clc
    fprintf('Processing file # %d of %d for streamlines\n',iFile,nFiles);
    hold on
    
    %% Plot velocity field (Modified to Streamlines)
    u_staggered = load(uFile(iFile).name);
    v_staggered = load(vFile(iFile).name);
    
    % Convert staggered velocity to cell center values
    % The 'uc' and 'vc' arrays are updated by the cellcenter function
    [uc, vc] = cellcenter(uc, vc, u_staggered, v_staggered, Nx, Ny);
    
    % --- MODIFICATION START: Contour plot replaced by Streamline plot ---
    % Original contour plot of u-component (commented out):
    % contourf(xu,yu,u_staggered,30,'edgecolor','none')
    % colorbar % Keep if needed for other plots, or remove if only for contourf
    % caxis([cmin, cmax]) % cmin, cmax calculation was commented out
    
    % New streamline plot using cell-centered velocities (uc, vc)
    % xp and yp are coordinates for cell centers (loaded before the loop)
    if exist('xp', 'var') && exist('yp', 'var')
        hh = streamslice(xp, yp, uc, vc, 10,'arrows'); % Density 1.5 is an example, adjust as needed
                                         % 'uc' and 'vc' are assumed to be correctly oriented 
                                         % with 'xp' and 'yp' (e.g. all Ny x Nx)
        set(hh, 'LineWidth',0.5,'Color','b','LineStyle','-');
        axis tight; % Adjust axis to data limits
    else
        warning('xp or yp mesh data not found. Cannot plot streamlines.');
    end
    % --- MODIFICATION END ---
   
    %    quiver(xp,yp,uc',vc'); % This was originally commented, keep as is or use for debugging
    
    %% Plot mesh (existing code, if needed)
%      mesh(xp,yp,0*xp,'FaceAlpha','0.0','EdgeColor','w','LineStyle','-','EdgeAlpha','0.20')
% view(90,0) % Example view

    axis equal % Important for correct aspect ratio of streamlines
    
    % Update title for streamlines
    if exist('nml','var') && isfield(nml,'time') && isfield(nml.time,'dt') && isfield(nml.time,'it_save')
        title_str = sprintf('Streamlines at t = %.3f (File %d)', iFile*nml.time.dt*nml.time.it_save, iFile);
    else
        title_str = sprintf('Streamlines (File %d)', iFile);
    end
    title(title_str);

    %% Plot FEM mesh (existing code)
    M = load("MP.txt"); % Ensure MP.txt is available
    P_data = load(PFile(iFile).name); % Renamed to avoid conflict with figure property P
    plot(polyshape(P_data(1,1:30),P_data(2,1:30)),'FaceColor',[0.93,0.69,0.13],'FaceAlpha',1);
    % Plot dividing line
    plot([Lx,Lx],[0,Ly/2],'r-','linewidth',10);
    plot([Lx,Lx],[Ly/2,Ly],'k-','linewidth',10);
    
    %% Plot cilia (existing code)
    C_data = load(CFile(iFile).name)'; % Renamed to avoid conflict
    for i = 1:2:size(C_data,2)
%         plot(C_data(:,i),C_data(:,i+1),'k.','linewidth',10); % Changed to black dots for better visibility with streamlines
        scatter(C_data(:,i),C_data(:,i+1),60,'k','filled');
    end
    
    xlim([4.5 6.5]*1e-3); % This was for a specific zoom, commented out in original
    % daspect([1,1,1]) % axis equal should handle this

    writeVideo(vid,getframe(gcf));
    if iFile ~= nFiles % Ensure figure is cleared for the next frame, unless it's the last one
%         clf
    end
end
close(vid);
disp('Finished generating video with streamlines: ibm_c_streamlines.avi');

% Function to convert staggered velocity to cell-centered velocity
% Assuming this function is correctly defined and works as intended
function [uc_out,vc_out] = cellcenter(uc_in_placeholder, vc_in_placeholder, u, v, Nx, Ny)
    % Initialize output arrays to ensure correct dimensions and clear previous values
    uc_out = zeros(Ny,Nx); 
    vc_out = zeros(Ny,Nx);

    % Calculate cell-centered vc
    for j_idx = 2:Nx+1 % Corresponds to x-index in staggered grid
        for i_idx = 1:Ny % Corresponds to y-index
            vc_out(i_idx, j_idx-1) = v(i_idx, j_idx); 
        end
    end

    % Calculate cell-centered uc
    for j_idx = 1:Nx % Corresponds to x-index
        for i_idx = 2:Ny+1 % Corresponds to y-index in staggered grid
            uc_out(i_idx-1, j_idx) = u(i_idx, j_idx); 
        end
    end
end
