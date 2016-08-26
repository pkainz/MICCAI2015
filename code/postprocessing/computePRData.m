function [ AUC_raw, AUC_filtered, AUC_interp, ...
    s_perf_data, ...
    filtered_data, ...
    interp_data ] = computePRData ( perf_data )
%% COMPUTEPRDATA plot the performance data for a precision/recall plot
% perf_data     the performance data matrix Nx11, each row contains a
%               threshold/score
% type          0 for classification, 1 for regression
% AUC_raw       the area under the curve (unfilterd data)
% AUC_filtered  the area under the curve (max-filterd data)
% AUC_interp    the area under the curve (interpolated data)
% s_perf_data   the raw performance data, sorted by recall
% filtered_data the max-filtered data matrix

% remove all NaN's from the precision and recall
nanindices = find(isnan(perf_data(:,3)) == 1);
perf_data(nanindices, :) = [];

% sort performance data by recall
s_perf_data = sortrows(perf_data, 2);
start = zeros(1,size(perf_data, 2));
start(3) = 1; % set the precision at recall 0 to 1.
s_perf_data = vertcat(start, s_perf_data(2:end,:));

%% interpolate the data
% get the max at each recall value
recalls = unique(s_perf_data(:,2));
filtered_data = zeros(length(recalls),2);
for recall_idx = 1 : length(recalls)
    recall_ = recalls(recall_idx);
    filtered_data(recall_idx, 1) = recall_;  
    max_precision_ = max(s_perf_data(s_perf_data(:,2) == recall_, 3));
    filtered_data(recall_idx, 2) = max_precision_;
end

% search for interpolated points
ctr2 = 1;
nextIndex = -1;
interp_data = zeros(1,2);
previous = [0 1]; % start at 0,1
for recall_idx = 2 : length(recalls) 
    % skip some elements
    if (recall_idx < nextIndex)
        continue;
    end
    
    %fprintf('recall_idx = %d\n', recall_idx);
    
    prev_recall_ = previous(1);
    prev_precision_ = previous(2);
    
    recall_ = filtered_data(recall_idx, 1);
    precision_ = filtered_data(recall_idx, 2);
    
    % starting from the current recall index
    % search for the next precision being equal or higher than the current
    next_max = max(...
        filtered_data(...
            filtered_data(:,1) >= recall_ &...
            filtered_data(:,2) >= precision_...
            , 2));
    prc_idx = find(filtered_data(:,2) == next_max, 1, 'last');
    
    max_precision_ = filtered_data(prc_idx, 2);
    
    % set the points
    interp_data(ctr2, 1) = prev_recall_;
    interp_data(ctr2, 2) = max_precision_;
    interp_data(ctr2+1, 1) = filtered_data(prc_idx, 1);
    interp_data(ctr2+1, 2) = max_precision_;
    
    % jump to the prc_idx 
    nextIndex = prc_idx+1;
    % remember the previous recall 
    previous = [filtered_data(prc_idx, 1), max_precision_];
    
    % increase counter
    ctr2 = ctr2 + 2;
end

% remove all information but recall and precision
s_perf_data = [s_perf_data(:,2), s_perf_data(:,3)];

%% compute the AUC
AUC_interp = trapz(interp_data(:,1), interp_data(:,2));
AUC_filtered = trapz(filtered_data(:,1), filtered_data(:,2));
AUC_raw = trapz(s_perf_data(:,1), s_perf_data(:,2));

return;
end
