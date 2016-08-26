%% this script reads in the centers of all images into a struct 'locations'
% list all txt files ending with "*.txt" and read the ground truth centers
txtFileList = filelist(path_center_images,'*.txt');

for i = 1 : size(txtFileList,1)
    fileName = txtFileList{i};
    fprintf('Reading center locations from %s\n',fileName);
    
    fileContent = dlmread(fileName, ' ', 1, 0);
    % get the centers
    centers = fileContent(:, 1:2);
    locations.([ 'img_', num2str(i) ]) = centers;
end

% remove all vars except for the locations structure
% clearvars fileContent c_x c_y centers fileName i txtFileList
