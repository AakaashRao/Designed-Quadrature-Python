clc;clear all;close all;
d=10;p=5;n_s=300;
tic
[XW,deltamain]=generator(d,p,n_s);
toc
filename = sprintf('TO_d_%d_accu_%d_node_%d.xlsx', d,p,n_s);
xlswrite(filename,XW);
%%First d columns of XW are the nodes and last column is the weights. 

%%%For Gaussian e.x. check \int_{\R^25} x_1^2 + x_2^2 \rho(x_1,...,x_25) dxdy = 2
%int_2_var=[XW(:,1).^2+XW(:,2).^2]'*XW(:,d+1);
