%Evaluate the gradients of the learnable parameters of the second neural
%network (parameter2)
function [gradients2,loss2,parameters2] = modelGradients_parameters2(parameters,parameters2,dlX,dlZ,dlT,S0IC)

%Make prediction of the coordinates of the free surface

% Calculate lossS. Enforce initial conditions.
dlS=model_2(parameters2,S0IC(1,:),S0IC(2,:));
lossS = mse(dlS, S0IC(3,:));

% Calculate lossS. Enforce the real condition
% i.e. assuming that the predicted value for the z-coordinate is accurate, 
% inputting this value into the first neural network, which has already been trained, 
% should yield the correct value for the head. Ideally, the difference between the output of the first neural network 
% and the predicted value should be zero.
dlS=model_2(parameters2,dlX,dlT);
dlZ=dlS;
H = model(parameters,dlX,dlS,dlT);
zeroTarget_diff = zeros(size(H), 'like', H);
h_diff=H-dlS;
loss_surface=mse(h_diff,zeroTarget_diff);


% Combine losses
loss2 = lossS + loss_surface;


% Calculate gradients with respect to the learnable parameters.
gradients2 = dlgradient(loss2,parameters2);

end