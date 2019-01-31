close all;
clear;
clc;

addpath('../data/dataset2');

%data
%{
img_name = '1';
img_format = '.jpg';
depth_format = '.pgm';
%}

%read rgbd data
%{
rgb = imread(strcat(img_name, img_format));
depth = imread(strcat(img_name, depth_format));
%}

%plot of the input
%{
figure(1);
subplot(1, 2, 1);
imshow(rgb);
title('Color input');
subplot(1, 2, 2);
imshow(depth);
title('Depth input');
%}

ptCloud = pcread('Person_Kinect.ply');
pcshow(ptCloud);

