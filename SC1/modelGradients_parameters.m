%Evaluate the gradients of the learnable parameters of the first neural
%network (parameter)
function [gradients,loss,parameters] = modelGradients_parameters(parameters,parameters2,dlX,dlZ,dlT,dlX0,dlZ0,dlT0,dlH0,S0IC,dlX0BC3,dlX0BC4,dlX0BC5,dlX0BC6,dlZ0BC3,dlZ0BC4,dlZ0BC5,dlZ0BC6,dlT0BC3,dlT0BC4,dlT0BC5,dlT0BC6,T,S)

% Make predictions with the collocation points using the first neural
% netwok (model)
H = model(parameters,dlX,dlZ,dlT);

% Evaluate derivatives with respect to X, Z and T for solving the ground
% water flow equation:

% ∂/∂x(K_xx(x,z)*h(x,z,t)*∂h/∂x(x,z,t))+∂/∂y(K_yy(x,z)*h(x,z,t)*∂h/∂y(x,z,t))=Sy∂h/∂t(x,y,t)+r 

gradientsH = dlgradient(sum(H,'all'),{dlX,dlZ,dlT},'EnableHigherDerivatives',true);
Hx = gradientsH{1};
Hz = gradientsH{2};
Ht = gradientsH{3};

dHx=dlgradient(sum(Hx,'all'),dlX,'EnableHigherDerivatives',true);
dHz=dlgradient(sum(Hz,'all'),dlZ,'EnableHigherDerivatives',true);

Tx=T.*dHx;
Tz=T.*dHz;
SH=S.*Ht;

% Calculate lossF. Enforce groundwater flow equation
f = Tx + Tz - SH;
zeroTarget = zeros(size(f), 'like', f);
lossF = mse(f, zeroTarget);

% Calculate lossU. Enforce initial and boundary conditions (Dirichelet and Neumann).
dlH0Pred = model(parameters,dlX0,dlZ0,dlT0); %Initial and Dirichelet conditions
lossU = mse(dlH0Pred, dlH0);

%Neumann conditions
dlH0Pred_IMP1 = model(parameters,dlX0BC3,dlZ0BC3,dlT0BC3);
gradients_Z = dlgradient(sum(dlH0Pred_IMP1,'all'),dlZ0BC3,'EnableHigherDerivatives',true);
zeroTarget_Z = zeros(size(gradients_Z), 'like', gradients_Z);
lossZ1 = mse(gradients_Z, zeroTarget_Z);

dlH0Pred_IMP2 = model(parameters,dlX0BC4,dlZ0BC4,dlT0BC4);
gradients_Z = dlgradient(sum(dlH0Pred_IMP2,'all'),dlZ0BC4,'EnableHigherDerivatives',true);
zeroTarget_Z = zeros(size(gradients_Z), 'like', gradients_Z);
lossZ2 = mse(gradients_Z, zeroTarget_Z);

dlH0Pred_IMP3 = model(parameters,dlX0BC5,dlZ0BC5,dlT0BC5);
gradients_X = dlgradient(sum(dlH0Pred_IMP3,'all'),dlX0BC5,'EnableHigherDerivatives',true);
zeroTarget_X = zeros(size(gradients_X), 'like', gradients_X);
lossX1 = mse(gradients_X, zeroTarget_X);

dlH0Pred_IMP4 = model(parameters,dlX0BC6,dlZ0BC6,dlT0BC6);
gradients_X = dlgradient(sum(dlH0Pred_IMP4,'all'),dlX0BC6,'EnableHigherDerivatives',true);
zeroTarget_X = zeros(size(gradients_X), 'like', gradients_X);
lossX2 = mse(gradients_X, zeroTarget_X);

% Combine losses
loss = lossF + lossU + lossZ1 +lossZ2 + lossX1 + lossX2;

% Calculate gradients with respect to the learnable parameters
gradients = dlgradient(loss,parameters);

end
