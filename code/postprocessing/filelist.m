function [ paths ] = filelist( directory, pattern )
%FILELIST Get the file list specified by a pattern in a directory
%   directory: the directory (absolute or relative) path without trailing
%   shlash
%   pattern: the file pattern, e.g. '*.tif'

files = dir( fullfile(directory, pattern) );
files = {files.name};

paths = cell(numel(files),1);

for i = 1 : numel(files)
    paths{i} = fullfile(directory, files{i});
end

return;
end

