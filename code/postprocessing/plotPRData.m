function [ plothandle ] = plotPRData( curves, type, varargin )
%PLOTPRDATA Plots a series of given PR curves, containing an averaged curve
% as last element in the array. 
% curves    given as 1D cell array

switch type
    case 0
        method = 'Class. Forest';
    case 1
        method = 'Regr. Forest';
    case 2
        method = 'Baseline';
    otherwise
        error('Unrecognized method type');
end

nCurves = numel(curves);
legendNames = cell(1,nCurves);

% get AUC measures, if provided
if length(varargin) == 1
    AUCs = cell2mat(varargin{1});
end

% PLOT THE ROC CURVES
plothandle = figure; hold on;
rnd = plot(0:0.2:1, 0:0.2:1, '--', 'color', [0.5 0.5 0.5]);
for i = 1 : nCurves
    pr_curve = curves{i};
    if ( i == nCurves ) 
        avgPlot = plot(pr_curve(:,1),pr_curve(:,2), ...
            'color', 'blue', 'linewidth', 3);
        
        if (length(varargin) == 1)
            legendName = { [method, ' avg.', ...
                ', AUC=', sprintf('%.3f', AUCs(end)) ]};
%     else
%         legendNames(i) = { ['Class', ' ', num2str(i) ] };
        end
    else
        plot(pr_curve(:,1),pr_curve(:,2), ...
            'color', [0.5 0.5 0.5], 'linewidth', 0.5);
    end
end
hold off;
xlim([0 1]), ylim([0 1.005]);
xlabel('recall'), ylabel('precision');
legend( [ avgPlot ], [ legendName ], 'location', 'southwest' );
title(sprintf('Precision/Recall Plot'));

return;
end

