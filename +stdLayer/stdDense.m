function [net, outputVar] = stdDense(net, inputVar, inputKSize, inputKNum, ...
    outputName, kNum, varargin)
% a standard dense layer

% parse params
opts.dropout = true;
opts.relu = true;
opts = vl_argparse(opts, varagin);
if numel(inputKSize) == 1, inputKSize = [inputKSize, inputKSize]; end

% add layer
net.addLayer(outputName, ...
    dagn.Conv('size', [inputKSize, inputKNum, kNum], ...
    'stride', 1, 'pad', 0), ...
    inputVar, ...
    [outputName, 'X'], ...
    {[outputName, 'Weights'], [outputName, 'Biases']});
inputVar = [outputName, 'X'];

% add relu layer as needed
if opts.relu
    net.addLayer([outputName, 'Relu'], ...
        dagnn.ReLU(), ...
        inputVar, ...
        [outputName, 'ReluX']);
    inputVar = [outputName, 'ReluX'];
end

% add dropout layer as needed
if opts.dropout
    opts.dropoutRate = 0.5;
    opts = vl_argparse(opts, varargin);
    net.addLayer([outputName, 'Dropout'], ...
        dagnn.DropOut('rate', opts.dropoutRate), ...
        inputVar, ...
        [outputName, 'DropoutX']);
    inputVar = [outputName, 'DropoutX'];
end

% get outputVar
outputVar = inputVar;

end
        