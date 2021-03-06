function net = stdLoss(net, inputVar, inputKNum, outputName, classNum, varargin)
% a standard loss layer

% parse params
opts.lossMethod = 'softmaxlog';

% add last conv layer
net.addLayer(outputName, ...
    dagnn.Conv('size', [1, 1, inputKNum, classNum]), ...
    inputVar, ...
    [outputName, 'X'], ...
    {[outputName, 'Weights'], [outputName, 'Biases']});

% add loss layer
net.addLayer([outputName, 'Loss'], ...
    dagnn.Loss('loss', opts.lossMethod), ...
    {[outputName, 'X'], 'label'}, ...
    [outputName, 'LossX']);

net.addLayer([outputName, 'AccuracyError'], ...
    dagnn.Loss('loss', 'classerror'), ...
    {[outputName, 'X'], 'label'}, ...
    [outputName, 'AccuracyErrorX']);


end