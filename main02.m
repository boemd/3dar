close all;
clear;
clc;

addpath('../toolbox');
% The path to the labeled dataset.
LABELED_DATASET_PATH = '..\data\dataset4\nyu_depth_v2_labeled.mat';

load(LABELED_DATASET_PATH, 'rawDepthFilenames', 'rawRgbFilenames');

%% Load a pair of frames 
load(LABELED_DATASET_PATH, 'images');
load(LABELED_DATASET_PATH, 'rawDepths');
%%
clc
img_idx=20;
alpha = 0.0028;
beta =300; %to arrange better
gamma = 1; %to arrange better
%ho scelto questo valore perchè la matrice Dx e Dy hanno valori che
%che stanno o stanno vicino a 0.01 o a 1x10^-4
%con questo valore tolgo praticamente i valori attorno a 1x10^-4 

%exstract the image and depth
imgRgb =images(:, :, :, img_idx);
imgDepthAbs = rawDepths(:, :, img_idx);
% Crop the images to include the areas where we have depth information.
%imgRgb = crop_image(imgRgb);
%imgDepthAbs = crop_image(imgDepthAbs);

%imgDepthFilled = fill_depth_cross_bf(imgRgb, double(imgDepthAbs));

%generate 3d points
points3d = rgb_plane2rgb_world(imgDepthAbs);

%print the image
figure(1);
%subplot(1, 2, 1);
imshow(imgRgb);
title('Color input');
hold on;
%subplot(1, 2, 2);
%k=histeq(imgDepthAbs, [0, 255]);
%imshow(k);
%title('Depth input');

%figure(2)
%pcshow(points3d)

%dimension of the image
[H, W] = size(imgDepthAbs);

%if problem throw error
assert(H == 480);
assert(W == 640);

%import camera params
camera_params;

%generate a number grid
[xx,yy] = meshgrid(1:W, 1:H);

%generate the 3 dimensional position in meter
X = (xx - cx_d) .* imgDepthAbs / fx_d;
Y = (yy - cy_d) .* imgDepthAbs / fy_d;
Z = imgDepthAbs;

%generate matrix with position and utility
pts = zeros(H, W, 6);
pts(:, :, 1) = X;  
pts(:, :, 2) = Y;  
pts(:, :, 3) = Z;  
pts(:, :, 4) = imgRgb(:, :, 1);  %r component
pts(:, :, 5) = imgRgb(:, :, 2);  %g component
pts(:, :, 6) = imgRgb(:, :, 3);  %b component

imgHsv = rgb2hsv(imgRgb);
pts2 = zeros(H, W, 4);
pts2(:, :, 1) = Z;  
pts2(:, :, 2) = imgHsv(:, :, 1);  %h component
pts2(:, :, 3) = imgHsv(:, :, 2);  %s component
pts2(:, :, 4) = imgHsv(:, :, 3);  %v component


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
f_DC = alpha.*(Pz.^2);
t_DC = gamma.*f_DC;
B = beta.*f_DC;

%find the two grandients
[Dx, Dy] = imgradientxy(Pz, 'intermediate');

%find the higth gradient(sono i bordi della nostra immagine)
idx1 = find((abs(Dx)-t_DC)>=0);
%idx1 = find((abs(Dx)-t_DC)>0);
C1 = zeros(H, W);
C1(idx1) = 1;

idx2 = find((abs(Dy)-t_DC)>=0);
%idx2 = find((abs(Dy)-t_DC)>0);
C2 = zeros(H, W);
C2(idx2) = 1;
%matrix that contain the position of the gradients
C = bitor(C1, C2);

[D,idx] = bwdist(C);

%T = zeros(H, W);

% for i = 1:H
%     for j=1:W
%         a= [X(i,j),Y(i,j), Z(i,j)];
%         %a= a(1:3);
%         b= [X(idx(i,j)),Y(idx(i,j)), Z(idx(i,j))];
%         %b= b(1:3);
%         T1(i, j) = norm(a - b);
%     end
% end
T=D./sqrt(2);

R = min(B, T);

normals = zeros(H,W, 3);
for i = 1:H
    for j=1:W
        if (Z(i,j) ~= 0)
            r = floor(R(i,j)*fx_d/Z(i,j));
        else
            r=0;
        end
        vp_h = [X(lim(i+r, 1, H), j)-X(lim(i-r, 1, H), j), Y(lim(i+r, 1, H), j)-Y(lim(i-r, 1, H), j), Sii(I_Pz, lim(i+1, 1, H), j, r-1, H, W)- Sii(I_Pz, lim(i-1, 1, H), j, r-1, H, W)]/2;
        vp_v = [X(i, lim(j+r, 1, W))-X(i, lim(j-r, 1, W)), Y(i, lim(j+r, 1, W))-Y(i, lim(j-r, 1, W)), Sii(I_Pz, i, lim(j+1, 1, W), r-1, H, W)- Sii(I_Pz, i, lim(j-1, 1, W), r-1, H, W)]/2;
        normals(i,j,:)=cross(vp_h,vp_v);
    end
end
Nx=normals(:, :, 1);
Ny=normals(:, :, 2);
Nz=normals(:, :, 3);
% figure(5)
% quiver3(X,Y,Z, Nx, Ny, Nz, 10 ,'MarkerEdgeColor', rand(1,3))
% hold off
%{
figure(2)
quiver3(X,Y,Z, normals(:, :, 1),normals(:, :, 2),normals(:, :, 3), 30)
hold on
view(-300,400)
axis([-2 2 -1 1 -.6 .6])
hold off

%}

norm_dist = zeros(480, 640, 4);
norm_dist(:,:,1)=Nx;
norm_dist(:,:,2)=Ny;
norm_dist(:,:,3)=Nz;
norm_dist(:,:,4)=Z;
norm_cols = reshape(normals, [], 3);
num_clusters = 3;
idxs = kmeans(norm_cols, num_clusters);
idxs = reshape(idxs, 480, 640, []);


figure(3)

for j=1:num_clusters
    %pts2 = reshape(pts2, [], 4);
    [t_r, t_c] = find(idxs==j);
    xxx=zeros(length(t_r),4);
    for l=1:length(t_r)
        xxx(l,:) = pts2(t_r(l), t_c(l),:);
    end    
    num_clusters2=3;
    idx2 = kmeans(xxx, num_clusters2);
    for k = 1:num_clusters2
        quiver3(X(idx2==k),Y(idx2==k),Z(idx2==k), Nx(idx2==k), Ny(idx2==k), Nz(idx2==k), 3, 'MarkerEdgeColor', rand(1,3))
        hold on
    end
end
% view(-300,400)
% axis([-2 2 -1 1 -.6 .6])
hold off

function s = Sii(Io, i, j, r, H, W)
    
    s = Io(lim(i+r, 1, H), lim(j+r, 1, W)) - Io(lim(i-r, 1, H), lim(j+r, 1, W)) - Io(lim(i+r, 1, H), lim(j-r, 1, W)) + Io(lim(i-r, 1, H), lim(j-r, 1, W));
    s = s/(4*r^2);
end

function v = lim(a, m, M)
    v = a;
    if (a < m)
        v = m;
    elseif (a > M)
        v = M;
    end    
end