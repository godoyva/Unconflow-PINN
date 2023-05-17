% Define a function that describe the first neural network(parameter)
function dlH = model(parameters,dlX,dlZ,dlT)

%The ANN works with three input variables and various layers
dlXZT = [dlX;dlZ;dlT];
numLayers = numel(fieldnames(parameters));

% First fully connect operation.
weights = parameters.fc1.Weights;
bias = parameters.fc1.Bias;
dlH = fullyconnect(dlXZT,weights,bias);

% Tanh and fully connect operations for remaining layers.
for i=2:numLayers
    name = "fc" + i;

    dlH = tanh(dlH);

    weights = parameters.(name).Weights;
    bias = parameters.(name).Bias;
    dlH = fullyconnect(dlH, weights, bias);
end
dlH = tanh(dlH);
end