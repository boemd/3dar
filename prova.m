close all;
clear;
clc;

A = zeros(2,2,2);
for i=1:2 
    for j=1:2 
        A(i,j,:)=[i;j];
    end
end

B = [1,2;1,2];
[r,c]=find(B==2);
