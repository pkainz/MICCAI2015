%% post processing script
% for MICCAI 2015 cell detection with RANDOM FORESTS

% run from console/bash script using
% matlab -nodisplay -nosplash -r "cd 'path/to/script', run('script.m'); exit;"
clear all; close all;

%% DATASET SETTINGS
% GRAZ = 0
% ICPR = 1;
dataset = 0;

%% METHOD SETTINGS
% class = 0
% regr = 1
method = 0;

%% general setup
% add Piotr's CV toolbox
addpath(genpath('../../code/toolbox/')); 
% add post processing scripts (computeDetectionPerformance.m, ...)
addpath(genpath('../../code/postprocessing/')); 

image_verbose = 1; % flag whether image results should be printed in detail
save_results = 1; % flag whether to store the results to the HDD
plotPRCurve = 1; % flag whether to plot the PR-curve (will be saved as fig and png as well)
computeBestThresholdStats = 1; % flag whether the threshold achieving the highest F1-score should be applied
% again and detailed performance statistics of the images are displayed

%% best performance settings
% the best performance will be shown as bounding boxes over the detections
show_nms_detection = 0; % show the detection results after applying the NMS

show_bb_img = 0; % show the resulting bounding boxes over the source images
printTP = 1; % print boxes for true positives
printFP = 1; % print boxes for false positives
printFN = 1; % print boxes for false negatives

if (method == 0)
    method_str = 'classification';
    method_str_short = 'class';
elseif (method == 1)
    method_str = 'regression';
    method_str_short = 'regr';
else
    warning('Unknown forest method, cannot run post-processing script!');
    return;
end

%% PATH TO THE EXPERIMENT ROOT
% GRAZ
if (dataset == 0)
    path_root = './bindata/'; 
    bb_size = 39; % size of the bounding box in the image overlay (just for plotting) (GRAZ: 39, ICPR: 11)
    dataset_str = 'GRAZ';
elseif (dataset == 1)
    % ICPR
    path_root = './bindata/';
    bb_size = 11; % size of the bounding box in the image overlay (just for plotting) (GRAZ: 39, ICPR: 11)
    dataset_str = 'ICPR';
else 
    warning('Unknown dataset, cannot run post-processing script!');
    return;
end

%% PATH TO THE SOURCES AND TARGETS
if (dataset == 0)
    % GRAZ data
    path_sources = '../../data/BM_GRAZ/source/';
    path_groundtruth = '../../data/BM_GRAZ/annotations/';
elseif (dataset == 1)
    % ICPR data
    path_sources = '../../data/ICPR/source/';
    path_groundtruth = '../data/ICPR/annotations/';
end

%% general path settings
path_predictions = strcat(path_root, 'predictions/');
path_results = strcat(path_root, 'results/');
path_figures = strcat(path_results, 'figures/');

% make result paths
mkdir(path_results);
mkdir(path_figures);

%% log the output of the script to a text file
if (save_results)
    path_log = strcat(path_results, 'log.txt');
    delete(path_log); % clear the log, if it exists
    diary(path_log); % make new log
end
fprintf('Running post-processing script for cell detection (%s FOREST)...\n',...
    upper(method_str));
fprintf('Started: %s\n', datestr(now));

%% do the post processing
% load file paths and get image names into a cell array
predImgFileList = filelist(path_predictions,'*.png');
nImages = length(predImgFileList);

if (nImages == 0)
    fprintf('No prediction images available for analysis.\n');
    fprintf('Finished: %s\n\n', datestr(now));
    if (save_results)
        diary off;
    end
    return;
end

%% general post processing settings
if (dataset == 0)
    distance_th = 10; % distance around a detected location (<=th)
    prediction_window = 50; % the window the prediction was produced with
    nms_radii=[7 7]; % neighbourhood for non-max suppression 
    use_border_extension = 1; % flag, whether the prediction images used border extension
    
    % some smoothing before non-max suppression
    gaussSigmas = 7.5; % 
    gaussRadius = 1; % 
    
elseif (dataset == 1)
    distance_th = 4; % distance around a detected location (<=th)
    prediction_window = 16; % the window the prediction was produced with 
    nms_radii=[3 3]; % neighbourhood for non-max suppression 
    use_border_extension = 1; % flag, whether the prediction images used border extension
    
    % some smoothing before non-max suppression
    gaussSigmas = 1.5; % 
    gaussRadius = 2; % 
    
end

% cover the entire 8bit space
threshold_start = 0;
threshold_step_size = 1;
threshold_end = 255;

nThresholds = length(threshold_start : threshold_step_size : threshold_end);

% collect plot data for P/R plot
% allocate the table for the performance data
% 1 = threshold
% 2 = recall
% 3 = precision
% 4 = average error
% 5 = F1-score
% 6 = FPR
% 7 = TPR
% 8 = accuracy
% 9 = std error
% 10= mean(abs(nGT-nTP))
% 11= std(abs(nGT-nTP))
perf_data = zeros(nThresholds, 11);
perf_idx = 1;

% full data cell array over all thresholds
thresholded_detection_results = cell(nThresholds,1);

% threshold at every possible grey value
for cth = threshold_start : threshold_step_size : threshold_end
    [ cth_struct ] = computeDetectionPerformance(...
        cth,...
        predImgFileList, ...
        path_groundtruth, ...
        prediction_window, ...
        use_border_extension, ...
        gaussSigmas, ...
        gaussRadius, ...
        nms_radii, ...
        distance_th, ...
        0, ... % nms-detection results are not shown during computation
        image_verbose);
    
    % fill the performance matrix for that threshold
    perf_data(perf_idx, :) = cth_struct.perf_data;
    
    % store the info in the cell array
    thresholded_detection_results{perf_idx} = cth_struct;
    
    perf_idx = perf_idx + 1;
end

% find the best index according to the best F1 score
best_perf_index = find(perf_data(:,5) == max(perf_data(:,5)));
% ensure, that only one gets selected on mutliple maximum answers
best_perf_index = best_perf_index(1);

%% get the values of the best threshold
best_threshold = round(perf_data(best_perf_index, 1)*255);
best_recall = perf_data(best_perf_index, 2);
best_precision = perf_data(best_perf_index, 3);
best_avg_distance_error = perf_data(best_perf_index, 4);
best_std_distance_error = perf_data(best_perf_index, 9);
best_f1_score = perf_data(best_perf_index, 5);
best_FPR = perf_data(best_perf_index, 6);
best_TPR = perf_data(best_perf_index, 7);
best_accuracy = perf_data(best_perf_index, 8);
best_avg_abs_diff_gt_tp = perf_data(best_perf_index, 10);
best_std_abs_diff_gt_tp = perf_data(best_perf_index, 11);

%% collect all results into a struct
all_results = struct;
all_results.info = strcat('Result data from ', upper(method_str), ' task [', path_predictions, ']');
all_results.date = datestr(now);
all_results.thresholded_detection_results = thresholded_detection_results;
all_results.nThresholds = length(thresholded_detection_results);
all_results.perf_data = perf_data;

all_results.best_perf_index = best_perf_index;
all_results.best_threshold = best_threshold;
all_results.best_recall = best_recall;
all_results.best_precision = best_precision;
all_results.best_avg_distance_error = best_avg_distance_error;
all_results.best_std_distance_error = best_std_distance_error;
all_results.best_f1_score = best_f1_score;
all_results.best_FPR = best_FPR;
all_results.best_TPR = best_TPR;
all_results.best_accuracy = best_accuracy;
all_results.best_avg_abs_diff_gt_tp = best_avg_abs_diff_gt_tp;
all_results.best_avg_std_diff_gt_tp = best_std_abs_diff_gt_tp;

all_results.nms_radii = nms_radii;
all_results.gaussSigmas = gaussSigmas;
all_results.gaussRadius = gaussRadius;
all_results.use_border_extension = use_border_extension;

all_results.path_sources = path_sources;
all_results.path_groundtruth = path_groundtruth;
all_results.path_predictions = path_predictions;

%% compute the PR curves for each image in order to visualize the variance
pr_collection = cell(1, nImages);
auc_collection = cell(1, nImages);
for img_i = 1 : nImages
    % gather all performance data for this image at all it's tested
    % thresholds
    raw_img_pr_data = zeros(nThresholds, 11);
    for th_i = 1 : nThresholds
        raw_img_pr_data(th_i, :) = thresholded_detection_results{th_i}.cth_image_stats{img_i}.perf_data;
    end
    
    % compute the performance data for each image
    [ ~, ~, AUC_interp, ...
        ~, ~, interp_pr_data ] = computePRData(raw_img_pr_data);
    
    % collect the data in a large array
    pr_collection{img_i} = interp_pr_data;
    auc_collection{img_i} = AUC_interp;
    
    %plotPRData({interp_pr_data}, method, {AUC_interp});
end

% add all perf data to all results
all_results.pr_collection = pr_collection;
all_results.auc_collection = auc_collection;

%% compute the average performance data for the forest
[ avg_AUC_raw, avg_AUC_filtered, avg_AUC_interp, ...
    avg_raw_pr_data, ...
    avg_filtered_pr_data, ...
    avg_interp_pr_data ] = computePRData(perf_data);

% add it to the total results
all_results.avg_AUC_raw = avg_AUC_raw;
all_results.avg_AUC_filtered = avg_AUC_filtered;
all_results.avg_AUC_interp = avg_AUC_interp;

all_results.avg_raw_pr_data = avg_raw_pr_data;
all_results.avg_filtered_pr_data = avg_filtered_pr_data;
all_results.avg_interp_pr_data = avg_interp_pr_data;

%% (optionally) save the overall data for that experiment
if (save_results)
    save(strcat(path_results, 'all_results.mat'), 'all_results');
end

%% plot the P/R curve
if (plotPRCurve)
    
    prPlot = plotPRData([pr_collection {avg_interp_pr_data}], method, {avg_AUC_interp});
    
    % save the figures to the figs directory
    if (save_results)
        saveas(prPlot, strcat(path_figures, method_str_short, '_pr-curve'), 'fig');
        saveas(prPlot, strcat(path_figures, method_str_short, '_pr-curve'), 'png');
    end
end

%% compute the best threshold statistics
if (computeBestThresholdStats)
    % always be verbose and plot the detection result in the best threshold
    [ best_th_struct ] = computeDetectionPerformance(...
        best_threshold,...
        predImgFileList, ...
        path_groundtruth, ...
        prediction_window, ...
        use_border_extension, ...
        gaussSigmas, ...
        gaussRadius, ...
        nms_radii, ...
        distance_th, ...
        show_nms_detection, ...
        1);
    
    % get the individual image statistics
    img_stats = best_th_struct.cth_image_stats;
    
    if (show_bb_img)
        % show all images
        for img_idx = 1 : nImages
            name = img_stats{img_idx}.name;
            
            detection_idcs_TP = img_stats{img_idx}.detection_idcs_TP;
            nTP = length(detection_idcs_TP);
            detection_idcs_FP = img_stats{img_idx}.detection_idcs_FP;
            nFP = length(detection_idcs_FP);
            detection_idcs_FN = img_stats{img_idx}.detection_idcs_FN;
            nFN = length(detection_idcs_FN);
            
            src = imread(strcat(path_sources, name, '.png'));
            % show the detections in the image
            bb_img_fig = figure(100+img_idx); clf; imshow(src), hold on;
            if (printTP)
                for tp_idx = 1 : nTP
                    idx = detection_idcs_TP(tp_idx);
                    [y, x] = ind2sub(size(src), idx);
                    %fprintf('detected TP cell at %d,%d\n', x, y);
                    rectangle('Position',[x-bb_size/2,y-bb_size/2,bb_size,bb_size], ...
                        'EdgeColor', 'green');
                end
            end
            
            if (printFP)
                for fp_idx = 1 : nFP
                    idx = detection_idcs_FP(fp_idx);
                    [y, x] = ind2sub(size(src), idx);
                    %fprintf('detected FP cell at %d,%d\n', x, y);
                    rectangle('Position',[x-bb_size/2,y-bb_size/2,bb_size,bb_size], ...
                        'EdgeColor', 'red');
                end
            end
            
            if (printFN)
                for fn_idx = 1 : nFN
                    idx = detection_idcs_FN(fn_idx);
                    [y, x] = ind2sub(size(src), idx);
                    %fprintf('missed cell at %d,%d\n', x, y);
                    rectangle('Position',[x-bb_size/2,y-bb_size/2,bb_size,bb_size], ...
                        'EdgeColor', 'yellow');
                end
            end
            
            title(sprintf([upper(method_str),' FOREST',...
                ' "%s"\n',...
                'hits (TP, green), wrong hits (FP, red), misses (FN, yellow)'],...
                strrep(name,'_','\_')));
            
            hold off;
            
            % optionally save the bounding box detections
            if (save_results)
                saveas(bb_img_fig, strcat(path_figures, method_str_short, 'bb_img_', name), 'fig');
                saveas(bb_img_fig, strcat(path_figures, method_str_short, 'bb_img_', name), 'png'); 
            end
        end
    end
end

% print out all results
all_results

% print the best performance results as table (for being copied) 
disptable(perf_data(best_perf_index, :), ['min_detection_threshold|recall|', ...
        'precision|avg_cell_distance_error|f1-score|FPR|TPR|accuracy|', ...
        'std_cell_distance_error|avg_abs_diff_gt_tp|std_abs_diff_gt_tp']);

fprintf('Finished: %s\n\n', datestr(now));

if (save_results) 
    % write the best achievable performance to a dedicated text file
    dlmwrite(strcat(path_results, 'best_results.txt'), ...
        perf_data(best_perf_index, :), ' ');
    % write the log file
    diary off;
end
