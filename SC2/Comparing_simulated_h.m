clc
close all
clear all

%load the obtained results
load results(90)_200epoche_35000cp.mat parameters

% Define X and Z nodes
x_node=20;
z_node=20;

%Import observed data
C=importdata("Observations_heterogeneous_anisotropic.xlsx");
data=C.data.Foglio1;
X=data(:,1);
Z=data(:,3);
H0=data(:,8);
H25=data(:,9);
H50=data(:,10);
H1=data(:,11);

% Mantain only active cells
idx0=find(H0==-888);
idx25=find(H25==-888);
idx50=find(H50==-888);
idx1=find(H1==-888);
H0(idx0)=nan;
H25(idx25)=nan;
H50(idx50)=nan;
H1(idx1)=nan;

% Deep learning array transform for coordinates
dlX=dlarray(X','CB');
dlZ=dlarray(Z','CB');

% Deep learning array transform for the different observed times
dlT0=dlarray(linspace(0.01,0.01,size(X,1)*1),'CB');
dlT25=dlarray(linspace(0.25,0.25,size(X,1)*1),'CB');
dlT50=dlarray(linspace(0.5,0.5,size(X,1)*1),'CB');
dlT1=dlarray(linspace(1,1,size(X,1)*1),'CB');

% Make prediction of the piezometric values using the first neural network
% (model --> parameters)
dlHPred0 = model(parameters,dlX,dlZ,dlT0);
dlHPred25 = model(parameters,dlX,dlZ,dlT25);
dlHPred50 = model(parameters,dlX,dlZ,dlT50);
dlHPred1 = model(parameters,dlX,dlZ,dlT1);

% Extract data from the deep learning array
HPred0=extractdata(dlHPred0);
HPred25=extractdata(dlHPred25);
HPred50=extractdata(dlHPred50);
HPred1=extractdata(dlHPred1);

% Mantain only the active cells in which the solution is available for the
% numerical model (observations)
HPred0(idx0)=nan;
HPred25(idx25)=nan;
HPred50(idx50)=nan;
HPred1(idx1)=nan;

% Contour Plot
levels=10;

contourf(reshape(X,x_node,z_node)',reshape(Z,x_node,z_node)',reshape(HPred0,x_node,z_node)',levels) %Predicted level t=0.01
figure
contourf(reshape(X,x_node,z_node)',reshape(Z,x_node,z_node)',reshape(H0,x_node,z_node)',levels) %Observed level t=0.01

figure
contourf(reshape(X,x_node,z_node)',reshape(Z,x_node,z_node)',reshape(HPred25,x_node,z_node)',levels) %Predicted level t=0.25
figure
contourf(reshape(X,x_node,z_node)',reshape(Z,x_node,z_node)',reshape(H25,x_node,z_node)',levels) %Observed level t=0.25

figure
contourf(reshape(X,x_node,z_node)',reshape(Z,x_node,z_node)',reshape(HPred50,x_node,z_node)',levels) %Predicted level t=0.5
figure
contourf(reshape(X,x_node,z_node)',reshape(Z,x_node,z_node)',reshape(H50,x_node,z_node)',levels) %Observed level t=0.50

figure
contourf(reshape(X,x_node,z_node)',reshape(Z,x_node,z_node)',reshape(HPred1,x_node,z_node)',levels) %Predicted level t=1
figure
contourf(reshape(X,x_node,z_node)',reshape(Z,x_node,z_node)',reshape(H1,x_node,z_node)',levels) %Observed level t=1

%Error plot
Err_t0=abs(reshape(H0,x_node,z_node)'-reshape(HPred0,x_node,z_node)');
Err_t025=abs(reshape(H25,x_node,z_node)'-reshape(HPred25,x_node,z_node)');
Err_t050=abs(reshape(H50,x_node,z_node)'-reshape(HPred50,x_node,z_node)');
Err_t1=abs(reshape(H1,x_node,z_node)'-reshape(HPred1,x_node,z_node)');

% Find the maximum error value to fix the superior lim
err_max0= max(Err_t0, [], 'all');
err_max025= max(Err_t025, [], 'all');
err_max050= max(Err_t050, [], 'all');
err_max1= max(Err_t1, [], 'all');
err_max=max([err_max1 err_max050 err_max025 err_max0]);

figure
imagesc(Err_t0);
colorbar
clim([0 err_max]) %max accepted and imposed value of error 0.03 (3%)

figure
imagesc(Err_t025);
colorbar
clim([0 err_max])

figure
imagesc(Err_t050);
colorbar
clim([0 err_max])

figure
imagesc(Err_t1);
colorbar
clim([0 err_max])

% Remove the nan values for evaluating the error between predicted and
% observed (simulated with the numerical model)
H0(idx0)=[];
H25(idx25)=[];
H50(idx50)=[];
H1(idx1)=[];

HPred0(idx0)=[];
HPred25(idx25)=[];
HPred50(idx50)=[];
HPred1(idx1)=[];

% Compute the MSE (m^2)
mse_t0=mse(HPred0',H0);
mse_t25=mse(HPred25',H25);
mse_t50=mse(HPred50',H50);
mse_t1=mse(HPred1',H1);

% Compute the RMSE (m)
RMSE_t0 = sqrt(mean((HPred0'-H0).^2));
RMSE_t25 = sqrt(mean((HPred25'-H25).^2));
RMSE_t50 = sqrt(mean((HPred50'-H50).^2));
RMSE_t1 = sqrt(mean((HPred1'-H1).^2));