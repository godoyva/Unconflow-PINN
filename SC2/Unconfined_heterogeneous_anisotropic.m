%Clear and close previously files, figures etc...
clear all
close all
clc

%Define the initial conditions and fixed heads 
fixhead_west=0.4;
fixhead_east=0.6;
starting_head=1;

%Import observed data (MODFLOW data)
C=importdata("Observations_heterogeneous_anisotropic.xlsx");
data=C.data.Foglio1; %C.data.nameofthespreasheet
X=data(:,1)'; %coordinate X
Z=data(:,3)'; %coordinate Z
H0=data(:,8)'; %heads for t=0.01
H25=data(:,9)'; %heads for t=0.25
H50=data(:,10)'; %heads for t=0.05
H1=data(:,11)'; %heads for t=1

%Provide only the coordinates of the active cells in the aquifer for each of the observed times (t=0.01, t=0.25, t=0.5, and t=1)
Xobs0=X;
Xobs25=X;
Xobs50=X;
Xobs1=X;
Zobs0=Z;
Zobs25=Z;
Zobs50=Z;
Zobs1=Z;
idx0=find(H0==-888);
idx25=find(H25==-888);
idx50=find(H50==-888);
idx1=find(H1==-888);
H0(idx0)=[];
H25(idx25)=[];
H50(idx50)=[];
H1(idx1)=[];
Xobs0(idx0)=[];
Xobs25(idx25)=[];
Xobs50(idx50)=[];
Xobs1(idx1)=[];
Zobs0(idx0)=[];
Zobs25(idx25)=[];
Zobs50(idx50)=[];
Zobs1(idx1)=[];

%Select a percentage of observed data to delete. This is to test the
%ability of the network to work with few data.
p = 0.8;
nSelect0 = round(numel(Xobs0)*p);
nSelect025 = round(numel(Xobs25)*p);
nSelect05 = round(numel(Xobs50)*p);
nSelect1 = round(numel(Xobs1)*p);
idx0 = randperm(numel(Xobs0),nSelect0);
idx25 = randperm(numel(Xobs25),nSelect025);
idx50 = randperm(numel(Xobs50),nSelect05);
idx1 = randperm(numel(Xobs1),nSelect1);
H0(idx0)=[];
H25(idx25)=[];
H50(idx50)=[];
H1(idx1)=[];
Xobs0(idx0)=[];
Xobs25(idx25)=[];
Xobs50(idx50)=[];
Xobs1(idx1)=[];
Zobs0(idx0)=[];
Zobs25(idx25)=[];
Zobs50(idx50)=[];
Zobs1(idx1)=[];

%Define the observed time vector. For each time, define the correspondig
%vector of size equal size to Xobs(t)
t=[0.01 0.25 0.5 1];
t0=repelem(t(1),size(Xobs0,2));
t25=repelem(t(2),size(Xobs25,2));
t50=repelem(t(3),size(Xobs50,2));
t1=repelem(t(4),size(Xobs1,2));

%Define Hydraulic parameters
T=0.001; %m/day
S=0.001;

%Define the number of Boundary Condition Points
numBoundaryConditionPoints_Fixed = 1000;
numBoundaryConditionPoints_Imp = 1000;

%Define the Boundary Conditions

%The west boundary fixed head condition
x0BC1 = zeros(1,numBoundaryConditionPoints_Fixed);
z0BC1 = linspace(0,fixhead_west,numBoundaryConditionPoints_Fixed);

%The east boundary fixed head condition
x0BC2 = ones(1,numBoundaryConditionPoints_Fixed);
z0BC2 = linspace(0,fixhead_east,numBoundaryConditionPoints_Fixed);

%The south boundary has an impermeable condition
x0BC3 = linspace(0,1,numBoundaryConditionPoints_Imp);
z0BC3 = zeros(1,numBoundaryConditionPoints_Imp);

%The north boundary has an impermeable condition
x0BC4 = linspace(0,1,numBoundaryConditionPoints_Imp);
z0BC4 = ones(1,numBoundaryConditionPoints_Imp);

%The west boundary has an impermeable condition above the fixed head
%condition
x0BC5 = zeros(1,numBoundaryConditionPoints_Imp);
z0BC5 = linspace(fixhead_west,1,numBoundaryConditionPoints_Imp);

%The east boundary has an impermeable condition above the fixed head
%condition
x0BC6 = ones(1,numBoundaryConditionPoints_Imp);
z0BC6 = linspace(fixhead_east,1,numBoundaryConditionPoints_Imp);

%Associate a time vector to each boundary condition that spans the investigated time range (t from 0 to 1)
t0BC1 = linspace(0,1,numBoundaryConditionPoints_Fixed);
t0BC2 = linspace(0,1,numBoundaryConditionPoints_Fixed);
t0BC3 = linspace(0,1,numBoundaryConditionPoints_Imp);
t0BC4 = linspace(0,1,numBoundaryConditionPoints_Imp);
t0BC5 = linspace(0,1,numBoundaryConditionPoints_Imp);
t0BC6 = linspace(0,1,numBoundaryConditionPoints_Imp);

%Define a vector of head values for the two conditions of fixed heads
u0BC1 = fixhead_west*ones(1,numBoundaryConditionPoints_Fixed);
u0BC2 = fixhead_east*ones(1,numBoundaryConditionPoints_Fixed);

%Define the number of initial condition points
numInitialConditionPoints  = 500;

%Define random coordinates
x0IC = rand([numInitialConditionPoints 1])';
z0IC = rand([numInitialConditionPoints 1])';

%Define the initial time
t0IC = zeros(1,(numInitialConditionPoints));

%Define the initial head value
u0IC = starting_head*ones(1,numInitialConditionPoints);

%Define the training data as if we were working with a classic ANN (initial
%conditions, fixed head conditions and observed head values)
X0 = [x0IC x0BC1 x0BC2 Xobs0 Xobs25 Xobs50 Xobs1];
Z0 = [z0IC z0BC1 z0BC2 Zobs0 Zobs25 Zobs50 Zobs1];
T0 = [t0IC t0BC1 t0BC2 t0 t25 t50 t1];
U0 = [u0IC u0BC1 u0BC2 H0 H25 H50 H1];

%Define the initial conditions for the free surface
S0IC=[x0IC;t0IC;u0IC];

%Define the number of internal collocation points
numInternalCollocationPoints = 35000;

%Use the function sobolset(#ofinputvariables)
pointSet = sobolset(3); %variables: x z t
points = net(pointSet,numInternalCollocationPoints);

%Define a vector for eache input variable
dataX = points(:,1);
dataZ = points(:,2);
dataT = points(:,3);

%Define an Array Datastore
ds = arrayDatastore([dataX dataZ dataT]);

%Architecture of the Network that will provide the head values
%Number of layers y number of neurons
numLayers = 9;
numNeurons = 20;

%Define a structure array
parameters = struct;
sz = [numNeurons 3]; %size of the input weight matrix

%Use the two functions "initializeHe" and "initializeZeros" to define the
%initial input weight matrix and the bias term
parameters.fc1.Weights = initializeHe(sz,2);
parameters.fc1.Bias = initializeZeros([numNeurons 1]);

%Define the initial values of the hidden weight matrices and the hidden
%bias terms
for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons]; %size of the hidden weight matrices
    numIn = numNeurons;
    parameters.(name).Weights = initializeHe(sz,numIn);
    parameters.(name).Bias = initializeZeros([numNeurons 1]);
end

%Define the output weight matrix and the output bias term
sz = [1 numNeurons]; %size of the output weight matrix
numIn = numNeurons;
parameters.("fc" + numLayers).Weights = initializeHe(sz,numIn);
parameters.("fc" + numLayers).Bias = initializeZeros([1 1]);

%Could show the implemented structure
parameters
parameters.fc1

% Architecture of the network that will provide the value of the z-coordinate
% for detecting the location of the free surface
numLayers = 9;
numNeurons = 20;

parameters2 = struct;

sz = [numNeurons 2]; %the second dimension is no longer 3, but 2. The input variables are only x and t.
parameters2.fc1.Weights = initializeHe(sz,2);
parameters2.fc1.Bias = initializeZeros([numNeurons 1]);

for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters2.(name).Weights = initializeHe(sz,numIn);
    parameters2.(name).Bias = initializeZeros([numNeurons 1]);
end

sz = [1 numNeurons];
numIn = numNeurons;
parameters2.("fc" + numLayers).Weights = initializeHe(sz,numIn);
parameters2.("fc" + numLayers).Bias = initializeZeros([1 1]);

%Just a control
parameters2
parameters2.fc1

%Training options
numEpochs = 200;
miniBatchSize = 128;
executionEnvironment = "cpu";
initialLearnRate = 0.01;
decayRate = 0.005;

%Deep learning array transform
mbq = minibatchqueue(ds, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFormat','BC', ...
    'OutputEnvironment',executionEnvironment);

dlX0 = dlarray(X0,'CB');
dlZ0 = dlarray(Z0,'CB');
dlT0 = dlarray(T0,'CB');
dlH0 = dlarray(U0);
dlX0BC3= dlarray(x0BC3,'CB');
dlX0BC4= dlarray(x0BC4,'CB');
dlX0BC5= dlarray(x0BC5,'CB');
dlX0BC6= dlarray(x0BC6,'CB');
dlZ0BC3= dlarray(z0BC3,'CB');
dlZ0BC4= dlarray(z0BC4,'CB');
dlZ0BC5= dlarray(z0BC5,'CB');
dlZ0BC6= dlarray(z0BC6,'CB');
dlT0BC3= dlarray(t0BC3,'CB');
dlT0BC4= dlarray(t0BC4,'CB');
dlT0BC5= dlarray(t0BC5,'CB');
dlT0BC6= dlarray(t0BC6,'CB');
S0IC=dlarray(S0IC,'CB');

% % Automaticcaly select the GPU environment
% if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
%     dlX0 = gpuArray(dlX0);
%     dlZ0 = gpuArray(dlZ0);
%     dlH0 = gpuArray(dlH0);
% end

%Define initial parameters for the correction weights procedure for the
%first neural network
averageGrad = [];
averageSqGrad = [];

%Optimize your training algorithm, for the first neural network, to increase its speed.
accfun = dlaccelerate(@modelGradients_parameters_eterog);

%Build a transient plot of the Loss Function
figure
C = colororder;
lineLoss = animatedline('Color',C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

%Start training for the first neural network (parameters)
start = tic;

iteration = 0;

for epoch = 1:numEpochs-100
    reset(mbq);

    while hasdata(mbq)
        iteration = iteration + 1;

        dlXZT = next(mbq);
        dlX = dlXZT(1,:);
        dlZ = dlXZT(2,:);
        dlT = dlXZT(3,:);

        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function (the modelGradients function now is accelerated and is called accfun).
        [gradients,loss,parameters] = dlfeval(accfun,parameters,parameters2,dlX,dlZ,dlT,dlX0,dlZ0,dlT0,dlH0,S0IC,dlX0BC3,dlX0BC4,dlX0BC5,dlX0BC6,dlZ0BC3,dlZ0BC4,dlZ0BC5,dlZ0BC6,dlT0BC3,dlT0BC4,dlT0BC5,dlT0BC6,T,S);

        % Update learning rate.
        learningRate = initialLearnRate / (1+decayRate*iteration);

        % Update the network parameters using the adamupdate function.
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);
    end

    % Plot training progress.
    loss = double(gather(extractdata(loss)));
    addpoints(lineLoss,iteration, loss);
    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
    drawnow
end

%Could you plot the state of your accelerated function
accfun


%Define initial parameters for the correction weights procedure for the
%second neural network
averageGrad2 = [];
averageSqGrad2 = [];

%Optimize your training algorithm, for the second neural network, to increase its speed.
accfun = dlaccelerate(@modelGradients_parameters2);

%Build a transient plot of the Loss Function
figure
C = colororder;
lineLoss = animatedline('Color',C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

%Start training for the second neural network (parameters2)
start = tic;

iteration = 0;

for epoch = 1:numEpochs
    reset(mbq);

    while hasdata(mbq)
        iteration = iteration + 1;

        dlXZT = next(mbq);
        dlX = dlXZT(1,:);
        dlZ = dlXZT(2,:);
        dlT = dlXZT(3,:);

        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function.
        [gradients2,loss,parameters2] = dlfeval(accfun,parameters,parameters2,dlX,dlZ,dlT,S0IC);

        % Update learning rate.
        learningRate2 = initialLearnRate / (1+decayRate*iteration);

        % Update the network parameters using the adamupdate function.
        [parameters2,averageGrad2,averageSqGrad2] = adamupdate(parameters2,gradients2,averageGrad2, ...
            averageSqGrad2,iteration,learningRate2);
    end

    % Plot training progress.
    loss = double(gather(extractdata(loss)));
    addpoints(lineLoss,iteration, loss);
    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
    drawnow
end

 accfun



%Define initial parameters for the correction weights procedure for the
%joint networks. 
averageGrad = [];
averageSqGrad = [];

%Optimize your training algorithm, for the joint neural network, to increase its speed.
accfun = dlaccelerate(@modelGradients_eterog);

%Build a transient plot of the Loss Function
figure
C = colororder;
lineLoss = animatedline('Color',C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

%Start training for the final joint neural network
start = tic;

for epoch = 1:numEpochs
    reset(mbq);

    while hasdata(mbq)
        iteration = iteration + 1;

        dlXZT = next(mbq);
        dlX = dlXZT(1,:);
        dlZ = dlXZT(2,:);
        dlT = dlXZT(3,:);

        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function.
        [gradients,loss,parameters_joint] = dlfeval(accfun,parameters,parameters2,dlX,dlZ,dlT,dlX0,dlZ0,dlT0,dlH0,S0IC,dlX0BC3,dlX0BC4,dlX0BC5,dlX0BC6,dlZ0BC3,dlZ0BC4,dlZ0BC5,dlZ0BC6,dlT0BC3,dlT0BC4,dlT0BC5,dlT0BC6,T,S);

        % Update learning rate.
        learningRate = initialLearnRate / (1+decayRate*iteration);

        % Update the network parameters using the adamupdate function.
        [parameters_joint,averageGrad,averageSqGrad] = adamupdate(parameters_joint,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);
        parameters2=parameters_joint(1,2);
        parameters=parameters_joint(1,1);
    end

    % Plot training progress.
    loss = double(gather(extractdata(loss)));
    addpoints(lineLoss,iteration, loss);
    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
    drawnow
end

 accfun


%Results

%Define a dataset of coordinates in which to estimate the free surface
XTest = linspace(0,1,100);
ZTest = linspace(0,1,100);
TTest0 = linspace(0,0,100);
TTest025 = linspace(0.25,0.25,100);
TTest05 = linspace(0.5,0.5,100);
TTest1 = linspace(1,1,100);

% Deep learning array transform
dlXTest = dlarray(XTest,'CB');
dlZTest = dlarray(ZTest,'CB');
dlTTest0 = dlarray(TTest0,'CB');
dlTTest025 = dlarray(TTest025,'CB');
dlTTest05 = dlarray(TTest05,'CB');
dlTTest1 = dlarray(TTest1,'CB');

% Make predictions using the neural network that is able to provide the
% z-coordinate of the free surface
dlS0=model_2(parameters2,dlXTest,dlTTest0);
dlS025=model_2(parameters2,dlXTest,dlTTest025);
dlS05=model_2(parameters2,dlXTest,dlTTest05);
dlS1=model_2(parameters2,dlXTest,dlTTest1);

%Transorm deep learning array in simple array
S0=extractdata(dlS0);
S025=extractdata(dlS025);
S05=extractdata(dlS05);
S1=extractdata(dlS1);

%Plot of the estimated free surface for different observed times
figure
area(S0)
figure
area(S025)
figure
area(S05)
figure
area(S1)

%Evaluate the values of the piezometric head below the estimated free surface

%For t=0.01
for i=1:size(dlS0,2)
    Z_grid0(i,:)=linspace(0,dlS0(i),50);
    Xgrid0(i,:)=repelem(XTest(i),50);
end
Z_grid0=extractdata(Z_grid0)';
Xgrid0=Xgrid0';
Z_grid0=reshape(Z_grid0,1,size(Z_grid0,1)*size(Z_grid0,2));
Xgrid0=reshape(Xgrid0,1,size(Xgrid0,1)*size(Xgrid0,2));
Tgrid0=linspace(0.01,0.01,size(Xgrid0,1)*size(Xgrid0,2));

%For t=0.25
for i=1:size(dlS025,2)
    Z_grid025(i,:)=linspace(0,dlS025(i),50);
    Xgrid025(i,:)=repelem(XTest(i),50);
end
Z_grid025=extractdata(Z_grid025)';
Xgrid025=Xgrid025';
Z_grid025=reshape(Z_grid025,1,size(Z_grid025,1)*size(Z_grid025,2));
Xgrid025=reshape(Xgrid025,1,size(Xgrid025,1)*size(Xgrid025,2));
Tgrid025=linspace(0.25,0.25,size(Xgrid025,1)*size(Xgrid025,2));

%For t=0.5
for i=1:size(dlS05,2)
    Z_grid05(i,:)=linspace(0,dlS05(i),50);
    Xgrid05(i,:)=repelem(XTest(i),50);
end
Z_grid05=extractdata(Z_grid05)';
Xgrid05=Xgrid05';
Z_grid05=reshape(Z_grid05,1,size(Z_grid05,1)*size(Z_grid05,2));
Xgrid05=reshape(Xgrid05,1,size(Xgrid05,1)*size(Xgrid05,2));
Tgrid05=linspace(0.5,0.5,size(Xgrid05,1)*size(Xgrid05,2));

%For t=1
for i=1:size(dlS1,2)
    Z_grid1(i,:)=linspace(0,dlS1(i),50);
    Xgrid1(i,:)=repelem(XTest(i),50);
end
Z_grid1=extractdata(Z_grid1)';
Xgrid1=Xgrid1';
Z_grid1=reshape(Z_grid1,1,size(Z_grid1,1)*size(Z_grid1,2));
Xgrid1=reshape(Xgrid1,1,size(Xgrid1,1)*size(Xgrid1,2));
Tgrid1=linspace(1,1,size(Xgrid1,1)*size(Xgrid1,2));

% Deep learning array transform
dlZgrid0=dlarray(Z_grid0,'CB');
dlXgrid0=dlarray(Xgrid0,'CB');
dlTgrid0=dlarray(Tgrid0,'CB');
dlZgrid025=dlarray(Z_grid025,'CB');
dlXgrid025=dlarray(Xgrid025,'CB');
dlTgrid025=dlarray(Tgrid025,'CB');
dlZgrid05=dlarray(Z_grid05,'CB');
dlXgrid05=dlarray(Xgrid05,'CB');
dlTgrid05=dlarray(Tgrid05,'CB');
dlZgrid1=dlarray(Z_grid1,'CB');
dlXgrid1=dlarray(Xgrid1,'CB');
dlTgrid1=dlarray(Tgrid1,'CB');

%Make prediction of the piezometric head values for different times
dlHPred0 = model(parameters,dlXgrid0,dlZgrid0,dlTgrid0);
dlHPred025 = model(parameters,dlXgrid025,dlZgrid025,dlTgrid025);
dlHPred05 = model(parameters,dlXgrid05,dlZgrid05,dlTgrid05);
dlHPred1 = model(parameters,dlXgrid1,dlZgrid1,dlTgrid1);

%Transorm deep learning array in simple array
HPred0=extractdata(dlHPred0);
HPred025=extractdata(dlHPred025);
HPred05=extractdata(dlHPred05);
HPred1=extractdata(dlHPred1);

%Flip your predictions
HPred0=flip(reshape(HPred0,50,100));
HPred025=flip(reshape(HPred025,50,100));
HPred05=flip(reshape(HPred05',50,100));
HPred1=flip(reshape(HPred1',50,100));

%Obtain the map of the coordinates
X_estimation0=flip(reshape(Xgrid0,50,100));
Z_estimation0=flip(reshape(Z_grid0,50,100));
X_estimation025=flip(reshape(Xgrid025,50,100));
Z_estimation025=flip(reshape(Z_grid025,50,100));
X_estimation05=flip(reshape(Xgrid05,50,100));
Z_estimation05=flip(reshape(Z_grid05,50,100));
X_estimation1=flip(reshape(Xgrid1,50,100));
Z_estimation1=flip(reshape(Z_grid1,50,100));