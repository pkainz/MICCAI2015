% read the locations from the txt files
fileList = filelist(path_center_images,'*.txt');
nImages = length(fileList);

% patchDir = '../patches/';
% mkdir(patchDir)

for i = 1 : nImages
    [ path, name, ext ] = fileparts(fileList{i});
    imgFileName = strcat(path_src_images, name,'.png');
    fprintf('%s\n',imgFileName);
    img = imread(imgFileName);
    
    height = size(img, 1);
    width = size(img, 2);
    
    target = zeros(height,width);
    sub_loc = locations.([ 'img_', num2str(i) ]); % c++ style indices!!!
    %%%%%%%%%%%%%%%
    % IMPORTANT!!
    sub_loc = sub_loc+1; % correct the indices to matlab style (+(1,1) offset)
    %%%%%%%%%%%%%%%
    lin_loc = sub2ind(size(target), sub_loc(:,2), sub_loc(:,1));
    target(lin_loc(:)) = 1;
    
    % arg 3 = patch size/cell size (single! scale)
    % arg 4 = alpha (controls shape of the exponential function)
    D_C = distanceTransform(target, 'exp', patch_size, alpha);
    
    dt.([ 'img_', num2str(i) ]) = D_C;
   
    % save the D_C as ground truth image
    imwrite(D_C, [path_target_images name '.png'], 'png');

    %% extract the patches (raw and ground truth)
    %[ imgPatches, gtPatches ] = cropPatches(img,D_C,sub_loc,patch_size);

    % save the patches
    %for p = 1 : length(imgPatches)
    %    imwrite(imgPatches{p}, [patchDir name '_' num2str(p) '_img.png'], 'png');
    %    imwrite(gtPatches{p}, [patchDir name '_' num2str(p) '_gtl.png'], 'png');
    %end
end


%% show the masked patch
% mask = dt.img_1 == 0;
% out = imoverlay(imread('HE_01.png'), mask, [0 0 0]);
% imshow(out);


