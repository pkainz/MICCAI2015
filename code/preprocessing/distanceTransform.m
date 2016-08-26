function [ D_C ] = distanceTransform( bw_img, method, varargin)
%DISTANCETRANSFORM Compute the distance transform (DT)
%   bw_img ... the binary image to compute the DT from 
%   method ... use 'exp' for the exponential DT used in Sironi CVPR'14 or
%   any other of the standard matlab bwdist methods
%   if the method is 'exp' one can pass several parameters
%       s ... Size of the neigbourhood (e.g. average cell size), default 21
%   alpha ... control constant for the decrease rate of d close to x,
%             default 3

D_C = zeros(size(bw_img));
fprintf('Using [%s] method for distance transform.\n', method);

if strcmp(method,'exp')
    % run through the image pixel by pixel
    % x             ... a point in the image
    % C             ... center points in the bw_img
    % D_C(x)        ... Euclidean distance transform of point x to NN in C
    % d(x)=-D_C(x)  ... inv. Euclidean DT
    
    if (isempty(varargin))
        s = 39;
    else 
        s = varargin{1};
    end
    % d_M           ... function of neighborhood size s thresholds d(x)
    d_M = s;
    % alpha         ... control constant for the decrease rate of d close to x
    if (length(varargin) < 2)
        alpha = 3;
    else
        alpha = varargin{2};
    end
    
%     figure;
    % compute the euclidean DT
    D_C = bwdist(bw_img, 'euclidean');
    %fprintf('EC_DT  min: %d, max: %d\n',min(D_C(:)), max(D_C(:)))
%     subplot(131), 
%     imshow(D_C, []);
    
    st_d_M = find(D_C<d_M);
    gt_d_M = find(D_C>=d_M);
    
    % use the exponential distance transform for each point (Sironi et al.,
    % CVPR'14)
    D_C(st_d_M) = exp(alpha*(1-(D_C(st_d_M)./d_M)))-1; 
    %fprintf('EX_DT  min: %d, max: %d\n',min(D_C(:)), max(D_C(:)))
%     subplot(132), 
%     imshow(D_C, []);
    
    % set the points where distance is greater-equal than d_M to zero
    D_C(gt_d_M) = 0;
    %fprintf('EX_DT2 min: %d, max: %d\n',min(D_C(:)), max(D_C(:)))
%     subplot(133), 
    %imshow(D_C, []);
    
    % scale it between 0 and 1
    D_C = mat2gray(D_C);
else
    D_C = mat2gray(-bwdist(bw_img, method));  
end

end

