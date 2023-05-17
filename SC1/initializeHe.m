%Define a function to randomly set initial value of the weight matrices
function parameter = initializeHe(sz,numIn,className)

arguments
    sz
    numIn
    className = 'single'
end

parameter = sqrt(2/numIn) * randn(sz,className);
parameter = dlarray(parameter);

end