%% extracts ground truth from dot-annotated images and writes to plaintext file 
%writes txt-files containing the ground truth locations of the annotated cells

% list all image files ending with "*.png"
imgFileList = filelist(path_center_images,'*.png');

if ~exist(path_target_images, 'file')
    mkdir(path_target_images)
end

for i = 1 : size(imgFileList,1)  
    % extract file information
    file = imgFileList{i}; 
    [path, name, ext] = fileparts(file);
    fprintf('reading %s\n', file);
    
    % read image
    I_ann = imread(file);
    
    % average the channels
    if (size(I_ann,3) ~= 1)
        I_ann = mean(I_ann, 3);
    end
    
    % find non-zero locations as annotation centers
    lins = find(I_ann ~= 0); % indices start at 1
    [rows, cols] = ind2sub(size(I_ann), lins);
    mat = [cols, rows] - 1; % shift the index -> to reflect starting by 0 (c++ style)
    
    % concat locations to matrix
    mat_ = [size(mat);mat];
    
    % write the matrix as whitespace delimited ASCII text file
    % first row contains the dimensions of the matrix
    dlmwrite(strcat(path_center_images,name,'.txt'), mat_, ' ');
    
    % print debug
    fprintf('Writing files for %s\n', strcat(path_target_images,name)); 
end

fprintf('Done.\n');