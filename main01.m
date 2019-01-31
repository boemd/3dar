close all;
clear;
clc;

addpath('../toolbox');
% The path to the labeled dataset.
LABELED_DATASET_PATH = '..\data\dataset4\nyu_depth_v2_labeled.mat';

load(LABELED_DATASET_PATH, 'rawDepthFilenames', 'rawRgbFilenames');

%% Load a pair of frames and align them.
load(LABELED_DATASET_PATH, 'images');
load(LABELED_DATASET_PATH, 'rawDepths');
%%
imgRgb =images(:, :, :, 5);
imgDepthAbs = rawDepths(:, :,5);
% Crop the images to include the areas where we have depth information.
%imgRgb = crop_image(imgRgb);
%imgDepthAbs = crop_image(imgDepthAbs);

%imgDepthFilled = fill_depth_cross_bf(imgRgb, double(imgDepthAbs));

points3d = rgb_plane2rgb_world(imgDepthAbs);
%%

figure(1);
%subplot(1, 2, 1);
imshow(imgRgb);
title('Color input');
%subplot(1, 2, 2);
%k=histeq(imgDepthAbs, [0, 255]);
%imshow(k);
%title('Depth input');

figure(2)
pcshow(points3d)

%%

[H, W] = size(imgDepthAbs);
assert(H == 480);
assert(W == 640);

camera_params;

[xx,yy] = meshgrid(1:W, 1:H);

X = (xx - cx_d) .* imgDepthAbs / fx_d;
Y = (yy - cy_d) .* imgDepthAbs / fy_d;
Z = imgDepthAbs;

pts = zeros(H, W, 6);

pts(:, :, 1) = X;  
pts(:, :, 2) = Y;  
pts(:, :, 3) = Z;  
pts(:, :, 4) = imgRgb(:, :, 1);  %r component
pts(:, :, 5) = imgRgb(:, :, 2);  %g component
pts(:, :, 6) = imgRgb(:, :, 3);  %b component


%%
Px = pts(:, :, 1);
Py = pts(:, :, 2);
Pz = pts(:, :, 3);

%integral images
I_Px = integralImage(Px);
I_Py = integralImage(Py);
I_Pz = integralImage(Pz);

%combinations of integral images
I_Pxx = I_Px .* I_Px;
I_Pxy = I_Px .* I_Py;
I_Pxz = I_Px .* I_Pz;
I_Pyy = I_Py .* I_Py;
I_Pyz = I_Py .* I_Pz;
I_Pzz = I_Pz .* I_Pz;

