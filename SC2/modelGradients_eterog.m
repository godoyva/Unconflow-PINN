%Evaluate the gradients of the learnable parameters of the two joint neural
%networks in the heterogeneous and anisotropic study case
function [gradients,loss,parameters_joint] = modelGradients_eterog(parameters,parameters2,dlX,dlZ,dlT,dlX0,dlZ0,dlT0,dlH0,S0IC,dlX0BC3,dlX0BC4,dlX0BC5,dlX0BC6,dlZ0BC3,dlZ0BC4,dlZ0BC5,dlZ0BC6,dlT0BC3,dlT0BC4,dlT0BC5,dlT0BC6,T,S)

% Make predictions with the collocation points using the first neural
% netwok (model)
H = model(parameters,dlX,dlZ,dlT);

% Evaluate derivatives with respect to X, Z and T for solving the ground
% water flow equation:

% ∂/∂x(K_xx(x,z)*∂h/∂x(x,z,t))+∂/∂y(K_zz(x,z)*∂h/∂y(x,z,t))=Sy∂h/∂t(x,y,t)+r 

gradientsH = dlgradient(sum(H,'all'),{dlX,dlZ,dlT},'EnableHigherDerivatives',true);
Hx = gradientsH{1};
Hz = gradientsH{2};
Ht = gradientsH{3};

%For the heterogeneous and anisotropic aquifer define the different value
%of the hydraulic conductivity
for i=1:size(H,2)
    if 0<=extractdata(dlX(i)) && extractdata(dlX(i))<=0.7 && 0<=extractdata(dlZ(i)) && extractdata(dlZ(i))<=0.4
        T_x=0.002;
        T_z=T_x*0.1;
    elseif 0.7<extractdata(dlX(i)) && extractdata(dlX(i))<=1 && 0<extractdata(dlZ(i)) && extractdata(dlZ(i))<=1
        T_x=0.003;
        T_z=T_x*0.1;
    elseif 0<extractdata(dlX(i)) && extractdata(dlX(i))<=0.4 && 0.4<extractdata(dlZ(i)) && extractdata(dlZ(i))<=1
        T_x=0.004;
        T_z=T_x*0.1;
    else
        T_x=0.001;
        T_z=T_x*0.1;
    end

    Tx_Hx(i)=T_x.*Hx(i);
    Tz_Hz(i)=T_z.*Hz(i);

end
SH=S.*Ht(i);
Tx=dlgradient(sum(Tx_Hx,'all'),dlX,'EnableHigherDerivatives',true);
Tz=dlgradient(sum(Tz_Hz,'all'),dlZ,'EnableHigherDerivatives',true);

% Calculate lossF. Enforce groundwater flow equation
f = Tx + Tz - SH;
zeroTarget = zeros(size(f), 'like', f);
lossF = mse(f, zeroTarget);

% Calculate lossU. Enforce initial and boundary conditions (Dirichelet and Neumann).
dlH0Pred = model(parameters,dlX0,dlZ0,dlT0);%Initial and Dirichelet conditions
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

%Make prediction of the coordinates of the free surface

% Calculate lossS. Enforce initial conditions.
dlS=model_2(parameters2,S0IC(1,:),S0IC(2,:));
lossS = mse(dlS, S0IC(3,:));

% Calculate lossS. Enforce the real condition
% i.e. assuming that the predicted value for the z-coordinate is accurate, 
% inputting this value into the first neural network, which has not already been trained, 
% should yield the uncorrect value for the head. Ideally, the difference between the output of the first neural network 
% and the predicted value should be zero.
dlS=model_2(parameters2,dlX,dlT);
dlZ=dlS;
H = model(parameters,dlX,dlS,dlT);
zeroTarget_diff = zeros(size(H), 'like', H);
h_diff=H-dlS;
loss_surface=mse(h_diff,zeroTarget_diff);


% Combine losses
loss =  lossF + lossU + lossZ1 +lossZ2 + lossX1 + lossX2 +lossS + loss_surface;

% Calculate gradients with respect to the learnable parameters
parameters_joint=[parameters,parameters2];
gradients = dlgradient(loss,parameters_joint);

end
