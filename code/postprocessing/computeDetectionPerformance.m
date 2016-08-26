function [ cth_struct ] = ...
    computeDetectionPerformance( ...
    cth, ...
    predImgFileList,...
    path_groundtruth, ...
    prediction_window, ... 
    use_border_extension, ...
    gaussSigmas, ...
    gaussRadius, ...
    nms_radii, ...
    distance_th, ...
    showNMSDetection, ...
    image_verbose)
%COMPUTEDETECTIONPERFORMANCE Computes the detection performance at a specific threshold for the non-max suppression.
%   Returns a struct, containing detailed as well as summarized information
%   on the postprocessed images.

nImages = length(predImgFileList);

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
perf_data = zeros(1, 11);

fprintf(strcat('#########################\n',...
    'Testing threshold %d\n'), cth);
minDetectionThreshold = cth/255;

% define the deviation (error) from the ground truth pixel of individual
% detections
cum_cell_dist_error = 0;
% the average of the errors over all images
%image_dist_errors = 0;

% allocate the cumulative confusion matrix (over all images)
% reality -> T   F
% test    T  TP  FP
%         F  FN  TN
cum_conf_mtrx = zeros(2,2);

% a counter for ground truth points which cannot be predicted
cum_nOutOfBounds = 0;

% overall counter for all detections
cum_nTotalDetections = 0;

% number of overall ground truth points
cum_nTotalGroundTruth = 0;

% container for all image stats (to be serialized to disk!)
cth_image_stats = cell(nImages,1);

% run through each of the images
for img_idx = 1 : nImages
    % confusion matrix for the individual image at a specific
    % nms threshold
    image_conf_mtrx = zeros(2,2);
    
    % localization error on the indiv. image
    image_distance_errors = zeros(0,1);
    
    % record the number of out of bounds ground truth points
    image_nOutOfBounds = 0;
    
    % overall counter for all detections
    image_nTotalDetections = 0;
    
    % number of overall ground truth points
    image_nTotalGroundTruth = 0;
    
    % record the performance data of the image
    image_perf_data = zeros(1,11);
    
    detection_idcs_TP = zeros(0,1);
    detection_idcs_FP = zeros(0,1);
    detection_idcs_FN = zeros(0,1);
    
    %% perform computations
    predFile = predImgFileList{img_idx};
    [path, name, ext] = fileparts(predFile);
    fprintf(strcat('===============================\n',...
        'Analyzing image %s (%2s%%)\n'), ...
        predFile, num2str(100*img_idx/nImages));
    
    % load the prediction image
    prediction = imread(predFile);
    imheight = size(prediction,1);
    imwidth = size(prediction,2);
    
    % load its ground truth
    ground_truth_locations = dlmread(...
        [path_groundtruth name '.txt'], ' ', 1, 0); % c++ style indices start at 0
    
     %%%%%%%%%%%%%%
    % IMPORTANT!!!
    % convert to matlab indices starting at 1!
    ground_truth_locations = ground_truth_locations+1;
    %%%%%%%%%%%%%%
    
    % remove all locations which cannot be predicted according
    % to the sliding window size (without border extension)
    gt_search_space = [ground_truth_locations(:,2),...
        ground_truth_locations(:,1)];
    
    % remove gt-points out of the detection boundaries
    if (use_border_extension == 0)
        gt_out_of_detection_bounds_y = find(...
            (gt_search_space(:,1) < round(prediction_window/2)) | ...
            (gt_search_space(:,1) > imheight - round(prediction_window/2)) ...
            );

        gt_search_space(gt_out_of_detection_bounds_y,:) = [];

        gt_out_of_detection_bounds_x = find(...
            (gt_search_space(:,2) < round(prediction_window/2)) | ...
            (gt_search_space(:,2) > imwidth - round(prediction_window/2)) ...
            );

        gt_search_space(gt_out_of_detection_bounds_x,:) = [];
        image_nOutOfBounds = ...
            (size(ground_truth_locations,1) - size(gt_search_space,1));
    end
    
    image_nTotalGroundTruth = size(ground_truth_locations,1)...
            - image_nOutOfBounds;
    
    % perform NMS using Piotr's CV toolbox
    I = double(prediction); % compute in double precision
    I = I/max(I(:)); % normalize the intensity range
    I = gaussSmooth(I,gaussSigmas,'same',gaussRadius); % smooth the image
    
    % if there is a border extension, crop the image after smoothing
    % such that the indices are OK again
    if (use_border_extension == 1)
       padding = floor(prediction_window/2);
       I = I(padding+1:end-padding, padding+1:end-padding);
       prediction = prediction(padding+1:end-padding, padding+1:end-padding);
    end
    
    %tic;
    % non-max suppression already returns indices sorted by confidence desc
    [subs2, vals2] = nonMaxSupr(I,nms_radii,minDetectionThreshold);
        
    %toc
    if (showNMSDetection)
        figure; clf; im(I); hold on;
        plot(gt_search_space(:,2), gt_search_space(:,1), 's',...
            'MarkerEdgeColor','r',...
            'MarkerFaceColor','g',...
            'MarkerSize', 10);
            
        plot(subs2(:,2),subs2(:,1),'ob');

        title(strrep(name,'_','\_'));
        legend('ground truth', 'detection hypotheses', ...
            'location', 'northoutside');
    end
    
    % create the detection image with the local maxima
    nms_img_grey = zeros(size(prediction));
    nms_img_grey(...
        sub2ind(size(prediction), subs2(:,1), subs2(:,2))) = ...
        255.*vals2;
    
    % determine total number of detections in the current image
    det_idcs_start = find(nms_img_grey~=0);
    image_nTotalDetections = length(det_idcs_start);
    
    % do k-nn search for each detection point (=query vector)
    % nn_gt_idx -> linear index of the nearest neighbour in the ground truth
    % search space
    % nn_gt_dist -> distance between each query and nearest neighbour
    % in the gt_search_space
    [nn_gt_idx, nn_gt_dist] = knnsearch(...
        gt_search_space, subs2, 'k', 1);
    
    % each GT can have multiple detections (however, we hope not!!)
    detections_per_gt = cell(image_nTotalGroundTruth,1);
        
    % run through each detection in the image
    % cdi is current detection index
    for cdi = 1 : image_nTotalDetections
        % get the confidence of the value
        curr_confidence = vals2(cdi);
        % get the distance of the current candidate
        curr_dist_error = nn_gt_dist(cdi);
        % get the linear index of the nn from the nn-vector
        curr_gt_idx = nn_gt_idx(cdi);
              
        %% let there just be the most confident detection
        % in the defined distance-threshold around the nearest ground truth
        
        % if detection is within the defined radius around the ground truth
        if (curr_dist_error <= distance_th) 
            
            % check, whether there is already a more confident detection 
            % for this GT point
            % if not, count as TP
            if (isempty(detections_per_gt{curr_gt_idx}))
                % TP
                % increase TP counter
                image_conf_mtrx(1,1) = image_conf_mtrx(1,1) + 1;

                % store the linear index of the detection in the TP indices
                detection_idcs_TP(image_conf_mtrx(1,1)) = sub2ind(...
                    size(prediction), subs2(cdi,1), subs2(cdi,2));

                % store the current distance error in a matrix (required for
                % assessing the standard deviation)
                image_distance_errors(image_conf_mtrx(1,1),1) = curr_dist_error;
            else
                % FP

                % increase FP counter
                image_conf_mtrx(1,2) = image_conf_mtrx(1,2) + 1;

                % store the linear index of the detection in the FP indices
                detection_idcs_FP(image_conf_mtrx(1,2)) = sub2ind(...
                    size(prediction), subs2(cdi,1), subs2(cdi,2));      
            end
            
            % record this hypothesis for a ground truth location
            % each row contains the index and the confidence of the
            % detection
            detections_per_gt{curr_gt_idx} = vertcat(...
                detections_per_gt{curr_gt_idx}, ...
                [   cdi, ...% the index of this detection (from sorted lists)
                    curr_confidence, ...% the score of this detection
                ]...
                );
        else
            % FP
            
            % increase FP counter
            image_conf_mtrx(1,2) = image_conf_mtrx(1,2) + 1;
            
            % store the linear index of the detection in the FP indices
            detection_idcs_FP(image_conf_mtrx(1,2)) = sub2ind(...
                size(prediction), subs2(cdi,1), subs2(cdi,2));
        end
    end
    
    %% collision detections (multiple detections per ground truth)
    gt_idcs_with_detections = find(cellfun(@isempty, detections_per_gt) == 0);
    if (showNMSDetection)
        for n_e_i = 1 : length(gt_idcs_with_detections)
            if (size(detections_per_gt{n_e_i},1) > 1)
               % get the ground truth location index
               gt_idx = n_e_i; 
               % re construct the ground truth point
               gt_x = gt_search_space(gt_idx, 2);
               gt_y = gt_search_space(gt_idx, 1);

               % found multiple detections for that gt  
               det_idcs = detections_per_gt{n_e_i}(:, 1); % detection indices in col 1

               for d_i = 1 : length(det_idcs)
                    det_x = subs2(det_idcs(d_i),2);
                    det_y = subs2(det_idcs(d_i),1);

                    % draw a line between all detections that vote vor this
                    % ground truth location
                    imline(gca, [gt_x, det_x], [gt_y, det_y]);
               end
            end
        end
    end
    
    % compute false negatives as number of all undetected ground truth points
    row_idcs_fn = find(cellfun(@isempty, detections_per_gt));
    image_conf_mtrx(2,1) = length(row_idcs_fn);
    detection_idcs_FN = sub2ind(size(prediction), ...
        gt_search_space(row_idcs_fn,1), gt_search_space(row_idcs_fn,2))';
        
    %% image performance stuff
    image_nTP = image_conf_mtrx(1,1);
    image_nFP = image_conf_mtrx(1,2);
    image_nFN = image_conf_mtrx(2,1);
    image_nTN = image_conf_mtrx(2,2);
    
    %% make sure some constraints are not violated
    assert(image_nTP <= image_nTotalGroundTruth);
    assert(image_nFP == (image_nTotalDetections - image_nTP));
    assert(image_nFN == (image_nTotalGroundTruth - image_nTP));
    
    % compute the performance data for this image
    image_recall = image_nTP / (image_nTP + image_nFN);
    image_precision = image_nTP / (image_nTP + image_nFP);
    image_TPR = image_recall;
    image_FPR = 1-image_precision;
    image_f1_score = 2*(image_precision*image_recall) / ...
        (image_precision+image_recall);
    
    % average distance error of all cells to their ground truth in this image
    image_avg_distance_error = sum(image_distance_errors) / image_nTP;
    image_std_distance_error = std(image_distance_errors);
    
    % compute the accuracy in this image
    image_accuracy = image_nTP/sum(image_conf_mtrx(:));
    
    % absolute difference between ground truth and true positives
    abs_diff_gt_tp = abs(image_nTotalGroundTruth-image_nTP);
    
    % fill the image's performance data
    image_perf_data(1) = cth/255;
    image_perf_data(2) = image_recall;
    image_perf_data(3) = image_precision;
    image_perf_data(4) = image_avg_distance_error;
    image_perf_data(5) = image_f1_score;
    image_perf_data(6) = image_FPR;
    image_perf_data(7) = image_TPR;
    image_perf_data(8) = image_accuracy;
    image_perf_data(9) = image_std_distance_error;
    image_perf_data(10)= abs_diff_gt_tp;
    % image_perf_data(11) is zero, since there is no std of one value
    
    % prepare writing image_stats to struct
    image_stats = struct;
    image_stats.cth = cth;
    image_stats.fullname = [name ext];
    image_stats.name = name;
    image_stats.ext = ext;
    image_stats.idx = img_idx;
    image_stats.nGT = image_nTotalGroundTruth;
    image_stats.nOutOfBounds = image_nOutOfBounds;
    image_stats.nDetections = image_nTotalDetections;
    % the absolute difference between the number of cells found and the
    % ground truth
    image_stats.abs_diff_gt_tp = abs_diff_gt_tp;
  
    image_stats.conf_mtrx = image_conf_mtrx;
    image_stats.detection_idcs_TP = detection_idcs_TP;
    image_stats.detection_idcs_FP = detection_idcs_FP;
    image_stats.detection_idcs_FN = detection_idcs_FN;
    
    % the average/std distance error in this image
    image_stats.avg_distance_error = image_avg_distance_error;
    image_stats.std_distance_error = image_std_distance_error;
    % save all individual distance errors in this image
    image_stats.image_distance_errors = image_distance_errors;
    
    image_stats.perf_data_descr = strcat('min_detection_threshold recall ', ...
        'precision avg_cell_distance_error f1-score FPR TPR accuracy ', ...
        'std_cell_distance_error');
    image_stats.perf_data = image_perf_data;
    image_stats.gaussSigmas = gaussSigmas;
    image_stats.gaussRadius = gaussRadius;
    image_stats.nms_radii = nms_radii;
    image_stats.minDetectionThreshold = minDetectionThreshold;
      
    % store the image in the current threshold
    cth_image_stats{img_idx} = image_stats;
    
    %% print some image stats
    if (image_verbose)
        fprintf('=== INDIVIDUAL IMAGE STATISTICS ===\n');
        image_conf_mtrx
        fprintf('Total detections: %d, total valid ground truth: %d\n', ...
            image_nTotalDetections, image_nTotalGroundTruth);
        fprintf('Ground truth out of bounds: %d (%f%%)\n', ...
            image_nOutOfBounds, 100*image_nOutOfBounds/image_nTotalGroundTruth);
        
        fprintf('Image accuracy: %f, distance error mean (SD): %f (%f)\n', ...
            image_accuracy, image_avg_distance_error, image_std_distance_error);
        fprintf('Recall: %f, Precision: %f, F1: %f\n', ...
            image_recall, image_precision, image_f1_score);
    end
    
    %% cumulation stuff
    % cumulate the total distance error over all images at that threshold
    cum_cell_dist_error = cum_cell_dist_error + sum(image_distance_errors);
    
    % cumulate the total # of hits/misses
    cum_conf_mtrx(1,1) = cum_conf_mtrx(1,1) + image_nTP;
    cum_conf_mtrx(1,2) = cum_conf_mtrx(1,2) + image_nFP;
    cum_conf_mtrx(2,1) = cum_conf_mtrx(2,1) + image_nFN;
    cum_conf_mtrx(2,2) = cum_conf_mtrx(2,2) + image_nTN;
    
    cum_nTotalDetections = cum_nTotalDetections + image_nTotalDetections;
    cum_nTotalGroundTruth = cum_nTotalGroundTruth + image_nTotalGroundTruth;
    cum_nOutOfBounds = cum_nOutOfBounds + image_nOutOfBounds;
end

%% compute the cumulative results over all images
% compute the performance stuff
nTP = cum_conf_mtrx(1,1);
nFP = cum_conf_mtrx(1,2);
nFN = cum_conf_mtrx(2,1);
nTN = cum_conf_mtrx(2,2);

recall = nTP / (nTP + nFN);
precision = nTP / (nTP + nFP);
TPR = recall;
FPR = 1-precision;
f1_score = 2*(precision*recall)/(precision+recall);

% compute the accuracy
accuracy = nTP/sum(cum_conf_mtrx(:));

% vector of all cell distances
all_cell_distance_errors = zeros(0,1);
% vector of averaged image errors
all_image_distance_errors = zeros(0,1);
% vector of abs(nGT-nTP) in the images
all_abs_diff_gt_tps = zeros(0,1);

fprintf('\n=============================\n');
% print out the summary
fprintf('=== IMAGE SUMMARY ===\n');
for file = 1 : nImages
    predFile = predImgFileList{file};
    [path_, name_, ext_] = fileparts(predFile);
    fprintf(['%s: Recall: %f, Precision: %f, F1: %f, ',...
        'distance error mean (SD): %f (%f), abs(nGT-nTP): %d\n'],...
        name_, ...
        cth_image_stats{file}.perf_data(2), cth_image_stats{file}.perf_data(3), ...
        cth_image_stats{file}.perf_data(5), cth_image_stats{file}.avg_distance_error, ...
        cth_image_stats{file}.std_distance_error, cth_image_stats{file}.abs_diff_gt_tp);
    
    % vertcat all individual cell errors
    all_cell_distance_errors = vertcat(all_cell_distance_errors, ...
        cth_image_stats{file}.image_distance_errors);
    
    % collect all distance errors
    all_image_distance_errors = vertcat(all_image_distance_errors, ...
        cth_image_stats{file}.avg_distance_error);
    
    % collect the absolute difference in ground truth vs true positives
    all_abs_diff_gt_tps = vertcat(all_abs_diff_gt_tps, ...
        cth_image_stats{file}.abs_diff_gt_tp);
end

% average the error over all individual true positive detections
avg_cell_distance_error = sum(all_cell_distance_errors) / nTP;
% compute the cell std-dev
std_cell_distance_error = std(all_cell_distance_errors);

% average image errors
avg_image_distance_error = sum(all_image_distance_errors) / nImages;
% compute the image std-dev
std_image_distance_error = std(all_image_distance_errors);

% average abs_diff_gt_tps
avg_abs_diff_gt_tps = sum(all_abs_diff_gt_tps) / nImages;
% compute the abs_diff_gt_tps 
std_abs_diff_gt_tps = std(all_abs_diff_gt_tps);

fprintf(['\n------------------------------------\n',...
    'Image(s) distance error mean (SD): %f (%f)\n'],...
        avg_image_distance_error, ...
        std_image_distance_error);

fprintf('\n=== CUMULATIVE STATISTICS ===\n');
cum_conf_mtrx
fprintf('Total detections: %d, total valid ground truth: %d\n', ...
    cum_nTotalDetections, cum_nTotalGroundTruth );
fprintf('Ground truth out of bounds: %d (%f%%)\n', ...
    cum_nOutOfBounds, 100*cum_nOutOfBounds/cum_nTotalGroundTruth );

fprintf('Overall accuracy: %f, distance error mean (SD): %f (%f), ', ...
    accuracy, avg_cell_distance_error, std_cell_distance_error);
fprintf('abs(nGT-nTP) mean (SD): %f (%f)\n', ...
    avg_abs_diff_gt_tps, std_abs_diff_gt_tps);
fprintf('Recall: %f, Precision: %f, F1: %f\n', ...
    recall, precision, f1_score);
fprintf('\n\n');

% fill the performance matrix for that threshold
perf_data(1) = cth/255;
perf_data(2) = recall;
perf_data(3) = precision;
perf_data(4) = avg_cell_distance_error;
perf_data(5) = f1_score;
perf_data(6) = FPR;
perf_data(7) = TPR;
perf_data(8) = accuracy;
perf_data(9) = std_cell_distance_error;
perf_data(10)= avg_abs_diff_gt_tps;
perf_data(11)= std_abs_diff_gt_tps;

% save all relevant information in this
% threshold to a struct and return it
cth_struct = struct;
cth_struct.cth = cth;
cth_struct.nImages = nImages;
cth_struct.minDetectionThreshold = minDetectionThreshold;
cth_struct.nGT = cum_nTotalGroundTruth;
cth_struct.nOutOfBounds = cum_nOutOfBounds;
cth_struct.nTotalDetections = cum_nTotalDetections;
cth_struct.conf_mtrx = cum_conf_mtrx;

cth_struct.avg_cell_distance_error = avg_cell_distance_error;
cth_struct.std_cell_distance_error = std_cell_distance_error;

cth_struct.avg_image_distance_error = avg_image_distance_error;
cth_struct.std_image_distance_error = std_image_distance_error;

cth_struct.avg_abs_diff_gt_tps = avg_abs_diff_gt_tps;
cth_struct.std_abs_diff_gt_tps = std_abs_diff_gt_tps;

cth_struct.perf_data_descr = strcat('min_detection_threshold recall ', ...
    'precision avg_cell_distance_error f1-score FPR TPR accuracy ', ...
    'std_cell_distance_error mean(abs(nGT-nTP)) std(nGT-nTP)');
cth_struct.perf_data = perf_data;
cth_struct.cth_image_stats = cth_image_stats;

cth_struct.nms_radii = nms_radii;
cth_struct.gaussSigmas = gaussSigmas;
cth_struct.gaussRadius = gaussRadius;
cth_struct.prediction_window = prediction_window;
cth_struct.use_border_extension = use_border_extension;

return;
end

