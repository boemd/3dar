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
img_idx=99;
alpha = 0.0028;
beta = 10;
gamma = 10;


imgRgb =images(:, :, :, img_idx);
imgDepthAbs = rawDepths(:, :, img_idx);
% Crop the images to include the areas where we have depth information.
%imgRgb = crop_image(imgRgb);
%imgDepthAbs = crop_image(imgDepthAbs);

%imgDepthFilled = fill_depth_cross_bf(imgRgb, double(imgDepthAbs));

points3d = rgb_plane2rgb_world(imgDepthAbs);
%

figure(1);
%subplot(1, 2, 1);
imshow(imgRgb);
title('Color input');
%subplot(1, 2, 2);
%k=histeq(imgDepthAbs, [0, 255]);
%imshow(k);
%title('Depth input');

%figure(2)
%pcshow(points3d)

%

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


%
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

% depth smoothing
f_DC = alpha.*Pz;
t_DC = gamma.*f_DC;
B = beta.*f_DC;

[Dx, Dy] = imgradientxy(Pz, 'intermediate');

idx1 = find((abs(Dx)-t_DC)>=0);
C1 = zeros(H, W);
C1(idx1) = 1;

idx2 = find((abs(Dy)-t_DC)>=0);
C2 = zeros(H, W);
C2(idx2) = 1;

C = C1 + C2;



[D,idx] = bwdist(C);

T = zeros(H, W);
for i = 1:H
    for j=1:W
        a= [X(i,j),Y(i,j), Z(i,j)];
        %a= a(1:3);
        b= [X(idx(i,j)),Y(idx(i,j)), Z(idx(i,j))];
        %b= b(1:3);
        T(i, j) = norm(a - b);
    end
end
T=T./sqrt(2);

R = min(B, T);

normals = zeros(H,W, 3);
for i = 1:H
    for j=1:W
        if (Z(i,j) ~= 0)
            r = floor(R(i,j)*fx_d/Z(i,j));
        else
            r=0;
        end
        vp_h = [X(i+r, j)-X(i-r, j), Y(i+r, j)-Y(i-r, j), Sii(I_Pz, i+1, j, r-1)- Sii(I_Pz, i-1, j, r-1)]/2;
        vp_v = [X(i, j+r)-X(i, j-r), Y(i, j+r)-Y(i, j-r), Sii(I_Pz, i, j+1, r-1)- Sii(I_Pz, i, j-1, r-1)]/2;
        normals(i,j)=cross(vp_h,vp_v);
    end
end




function s = Sii(Io, i, j, r)
    s = Io(i+r, j + r) - Io(i-r, j+r) - Io(i+r, j-r) + Io(i-r, j-r);
    s = s/(4*r^2);
end