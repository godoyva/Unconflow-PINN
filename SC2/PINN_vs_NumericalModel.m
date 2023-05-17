clear all
close all
clc
load results(90)_200epoche_35000cp.mat parameters parameters2
%Import observed data
C=importdata("Observations_heterogeneous_anisotropic.xlsx");
data=C.data.Foglio1;
X=data(:,1);
Z=data(:,3);
H0=reshape(data(:,8),20,20)';
H25=reshape(data(:,9),20,20)';
H50=reshape(data(:,10),20,20)';
H1=reshape(data(:,11),20,20)';
%Results
discr=200; %select the discretization for the PINN
%Define a dataset of coordinates in which to estimate the free surface
XTest = linspace(0,1,discr);
ZTest = linspace(0,1,discr);
TTest00=linspace(0,0,discr);
TTest0 = linspace(0.01,0.01,discr);
TTest025 = linspace(0.25,0.25,discr);
TTest05 = linspace(0.5,0.5,discr);
TTest1 = linspace(1,1,discr);

% Deep learning array transform
dlXTest = dlarray(XTest,'CB');
dlZTest = dlarray(ZTest,'CB');
dlTTest00 = dlarray(TTest00,'CB');
dlTTest0 = dlarray(TTest0,'CB');
dlTTest025 = dlarray(TTest025,'CB');
dlTTest05 = dlarray(TTest05,'CB');
dlTTest1 = dlarray(TTest1,'CB');

%Evaluate the values of the piezometric head below the estimated free surface

for i=1:size(dlZTest,2)
    Z_grid(i,:)=linspace(0,1,discr);
    Xgrid(i,:)=repelem(XTest(i),discr);
end
Z_grid=(Z_grid)';
Xgrid=Xgrid';
Z_grid=reshape(Z_grid,1,size(Z_grid,1)*size(Z_grid,2));
Xgrid=reshape(Xgrid,1,size(Xgrid,1)*size(Xgrid,2));

Tgrid0=linspace(0.01,0.01,size(Xgrid,1)*size(Xgrid,2));
Tgrid025=linspace(0.25,0.25,size(Xgrid,1)*size(Xgrid,2));
Tgrid05=linspace(0.5,0.5,size(Xgrid,1)*size(Xgrid,2));
Tgrid1=linspace(1,1,size(Xgrid,1)*size(Xgrid,2));


% Deep learning array transform
dlZgrid=dlarray(Z_grid,'CB');
dlXgrid=dlarray(Xgrid,'CB');
dlTgrid0=dlarray(Tgrid0,'CB');
dlTgrid025=dlarray(Tgrid025,'CB');
dlTgrid05=dlarray(Tgrid05,'CB');
dlTgrid1=dlarray(Tgrid1,'CB');

%Make prediction of the piezometric head values for different times
dlHPred0 = model(parameters,dlXgrid,dlZgrid,dlTgrid0);
dlHPred025 = model(parameters,dlXgrid,dlZgrid,dlTgrid025);
dlHPred05 = model(parameters,dlXgrid,dlZgrid,dlTgrid05);
dlHPred1 = model(parameters,dlXgrid,dlZgrid,dlTgrid1);

%Transorm deep learning array in simple array
HPred0=extractdata(dlHPred0);
HPred025=extractdata(dlHPred025);
HPred05=extractdata(dlHPred05);
HPred1=extractdata(dlHPred1);

% Make predictions using the neural network that is able to provide the
% z-coordinate of the free surface
dlS0=model_2(parameters2,dlXTest,dlTTest00);
dlS025=model_2(parameters2,dlXTest,dlTTest025);
dlS05=model_2(parameters2,dlXTest,dlTTest05);
dlS1=model_2(parameters2,dlXTest,dlTTest1);

%Transorm deep learning array in simple array
S0=extractdata(dlS0);
S025=extractdata(dlS025);
S05=extractdata(dlS05);
S1=extractdata(dlS1);

%Flip your predictions
HPred0_flip=flip(reshape(HPred0,discr,discr));
HPred025_flip=flip(reshape(HPred025,discr,discr));
HPred05_flip=flip(reshape(HPred05',discr,discr));
HPred1_flip=flip(reshape(HPred1',discr,discr));

%Obtain the map of the coordinates
X_estimation0=flip(reshape(Xgrid,discr,discr));
Z_estimation0=flip(reshape(Z_grid,discr,discr));


%estimation of the free surface
idx_close0 = 200-round(S0.*discr);
idx_close025 = 200-round(S025.*discr);
idx_close05 = 200-round(S05.*discr);
idx_close1 = 200-round(S1.*discr);
for i = 1:discr
    HPred0_flip(1:idx_close0(i), i) = NaN;
    HPred025_flip(1:idx_close025(i), i) = NaN;
    HPred05_flip(1:idx_close05(i), i) = NaN;
    HPred1_flip(1:idx_close1(i), i) = NaN;
end

%PINN solutions
axes=0:0.1:1;
axes=round(axes*discr);

% set the resolution to 300dpi
dpi = 300;
% set the file format to tif
format = '-dtiff';
%Current folder
filepath = pwd;

figure
imagesc(HPred0_flip);
cmap = colormap(jet);
cmap(1,:) = [1 1 1];
colormap(cmap);
clim([0.3, 1]);

% add colorbar and axis labels
c = colorbar;
c.Ticks = linspace(0.4, 1, 5);
xticks(axes);
xticklabels({'','0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1'});
yticks(axes);
yticklabels({'1','0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1','0'});
xlabel('X-axis');
ylabel('Y-axis');
ax = gca;  % get current axes handle
ax.FontName = 'Times New Roman';  % set font name
ax.FontSize = 12;  % set font size
ax.TickDir= 'out';
box off

% set the filename for the saved image
filename = 'piezoPINN_t0_eter';

% save the image
print(gcf, fullfile(filepath, filename), format, sprintf('-r%d',dpi));

figure
imagesc(HPred025_flip);
cmap = colormap(jet);
cmap(1,:) = [1 1 1];
colormap(cmap);
clim([0.37, 0.6]);

% add colorbar and axis labels
c = colorbar;
c.Ticks = linspace(0.4, 0.6, 5);
xticks(axes);
xticklabels({'','0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1'});
yticks(axes);
yticklabels({'1','0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1','0'});
% xlabel('X-axis');
ylabel('Y-axis');
ax = gca;  % get current axes handle
ax.FontName = 'Times New Roman';  % set font name
ax.FontSize = 12;  % set font size
ax.TickDir= 'out';
box off

% set the filename for the saved image
filename = 'piezoPINN_t25_eter';

% save the image
print(gcf, fullfile(filepath, filename), format, sprintf('-r%d',dpi));

figure
imagesc(HPred05_flip);
cmap = colormap(jet);
cmap(1,:) = [1 1 1];
colormap(cmap);
clim([0.37, 0.6]);

% add colorbar and axis labels
c = colorbar;
c.Ticks = linspace(0.4, 0.6, 5);
xticks(axes);
xticklabels({'','0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1'});
yticks(axes);
yticklabels({'1','0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1','0'});
xlabel('X-axis');
ylabel('Y-axis');
ax = gca;  % get current axes handle
ax.FontName = 'Times New Roman';  % set font name
ax.FontSize = 12;  % set font size
ax.TickDir= 'out';
box off

% set the filename for the saved image
filename = 'piezoPINN_t50_eter';

% save the image
print(gcf, fullfile(filepath, filename), format, sprintf('-r%d',dpi));

figure
imagesc(HPred1_flip);
cmap = colormap(jet);
cmap(1,:) = [1 1 1];
colormap(cmap);
clim([0.37, 0.6]);

% add colorbar and axis labels
c = colorbar;
c.Ticks = linspace(0.4, 0.6, 5);
xticks(axes);
xticklabels({'','0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1'});
yticks(axes);
yticklabels({'1','0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1','0'});
xlabel('X-axis');
ylabel('Y-axis');
ax = gca;  % get current axes handle
ax.FontName = 'Times New Roman';  % set font name
ax.FontSize = 12;  % set font size
ax.TickDir= 'out';
box off

% set the filename for the saved image
filename = 'piezoPINN_t1_eter';

% save the image
print(gcf, fullfile(filepath, filename), format, sprintf('-r%d',dpi));

%MODFLOW solutions
axes=0:0.1:1;
axes=round(axes*20);

figure
imagesc(H0);
cmap = colormap(jet);
cmap(1,:) = [1 1 1];
colormap(cmap);
clim([0.395, 1]);

% add colorbar and axis labels
c = colorbar;
c.Ticks = linspace(0.4, 1, 5);
xticks(axes);
xticklabels({'','0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1'});
yticks(axes);
yticklabels({'1','0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1','0'});
xlabel('X-axis');
ylabel('Y-axis');
ax = gca;  % get current axes handle
ax.FontName = 'Times New Roman';  % set font name
ax.FontSize = 12;  % set font size
ax.TickDir= 'out';
box off

% set the filename for the saved image
filename = 'piezoGMS_t0_eter';

% save the image
print(gcf, fullfile(filepath, filename), format, sprintf('-r%d',dpi));

figure
imagesc(H25);
cmap = colormap(jet);
cmap(1,:) = [1 1 1];
colormap(cmap);
clim([0.395, 0.6]);

% add colorbar and axis labels
c = colorbar;
c.Ticks = linspace(0.4, 0.6, 5);
xticks(axes);
xticklabels({'','0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1'});
yticks(axes);
yticklabels({'1','0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1','0'});
xlabel('X-axis');
ylabel('Y-axis');
ax = gca;  % get current axes handle
ax.FontName = 'Times New Roman';  % set font name
ax.FontSize = 12;  % set font size
ax.TickDir= 'out';
box off

% set the filename for the saved image
filename = 'piezoGMS_t25_eter';

% save the image
print(gcf, fullfile(filepath, filename), format, sprintf('-r%d',dpi));

figure
imagesc(H50);
cmap = colormap(jet);
cmap(1,:) = [1 1 1];
colormap(cmap);
clim([0.395, 0.6]);

% add colorbar and axis labels
c = colorbar;
c.Ticks = linspace(0.4, 0.6, 5);
xticks(axes);
xticklabels({'','0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1'});
yticks(axes);
yticklabels({'1','0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1','0'});
xlabel('X-axis');
ylabel('Y-axis');
ax = gca;  % get current axes handle
ax.FontName = 'Times New Roman';  % set font name
ax.FontSize = 12;  % set font size
ax.TickDir= 'out';
box off

% set the filename for the saved image
filename = 'piezoGMS_t50_eter';

% save the image
print(gcf, fullfile(filepath, filename), format, sprintf('-r%d',dpi));

figure
imagesc(H1);
cmap = colormap(jet);
cmap(1,:) = [1 1 1];
colormap(cmap);
clim([0.395, 0.6]);

% add colorbar and axis labels
c = colorbar;
c.Ticks = linspace(0.4, 0.6, 5);
xticks(axes);
xticklabels({'','0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9','1'});
yticks(axes);
yticklabels({'1','0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1','0'});
xlabel('X-axis');
ylabel('Y-axis');
ax = gca;  % get current axes handle
ax.FontName = 'Times New Roman';  % set font name
ax.FontSize = 12;  % set font size
ax.TickDir= 'out';
box off

% set the filename for the saved image
filename = 'piezoGMS_t1_eter';

% save the image
print(gcf, fullfile(filepath, filename), format, sprintf('-r%d',dpi));

