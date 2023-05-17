% Define a function that describe the second neural network(parameter)
function dlS = model_2(parameters2,dlX,dlT)

%The ANN works with two input variables and various layers
dlXT = [dlX;dlT];
numLayers = numel(fieldnames(parameters2));

% First fully connect operation.
weights = parameters2.fc1.Weights;
bias = parameters2.fc1.Bias;
dlS = fullyconnect(dlXT,weights,bias);

% Tanh and fully connect operations for remaining layers.
for i=2:numLayers
    name = "fc" + i;

    dlS = tanh(dlS);

    weights = parameters2.(name).Weights;
    bias = parameters2.(name).Bias;
    dlS = fullyconnect(dlS, weights, bias);
end
dlS = tanh(dlS);
end