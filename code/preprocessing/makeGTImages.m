clear all; close all;

% define the paths to the images
path_src_images = '../../data/BM_GRAZ/source/';
path_center_images = '../../data/BM_GRAZ/annotations/';
path_target_images = '../../data/BM_GRAZ/target/';

% initialization
patch_size = 39; % reflects average cell size (d_M)
% shape parameter for distance transform
alpha = 3; 

% optionally write GT locations
writeGTLocations;

% read (back in) the locations
readGTLocations;

% compute and extract the target labels for foreground
computeDistTrans;


