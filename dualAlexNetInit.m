function net = dualAlexNetInit(varargin)
% Initiliza a dual alexnet

% init opts
opts.initBias = 0.1;
opts.weightDecay = 1;
opts.imageDim = 1;
opts.batchNorm = true;
opts = vl_argparse(opts, varargin);

% create a empty net
net = dagnn.DagNN();
net.meta.inputSize = {[45 45 1 1], [79 79 1 1]};

% create network
% conv1 - cilia
[net, outputVar] = stdLayer.stdConv(net, 'ciliaInput', ...
    opts.imageDim, 'ciliaConv1', 11, 48, 'batchNorm', opts.batchNorm);
% conv2 - cilia
[net, outputVar] = stdLayer.stdConv(net, outputVar, ...
    48, 'ciliaConv2', 5, 64, 'pad', 2, 'batchNorm', opts.batchNorm);
% conv1 - image
[net, outputVar1] = stdLayer.stdConv(net, 'imageInput', ...
    opts.imageDim, 'imageConv1', 11, 48, 'stride', [2,2], ...
    'batchNorm', opts.batchNorm);
[net, outputVar1] = stdLayer.stdConv(net, outputVar1, ...
    48, 'imageConv2', 5, 64, 'pad', 2, 'batchNorm', opts.batchNorm);
% merge and conv3
[net, outputVar] = stdLayer.stdMerge(net, ...
    {outputVar, outputVar1}, 'convMerge');
[net, outputVar] = stdLayer.stdConv(net, outputVar, ...
    128, 'Conv3', 3, 96, 'pooling', false, ...
    'pad', 1, 'batchNorm', opts.batchNorm);
% conv4
[net, outputVar] = stdLayer.stdConv(net, outputVar, ...
    96, 'Conv4', 3, 96, 'pooling', false, ...
    'pad', 1, 'batchNorm', opts.batchNorm);
% conv5
[net, outputVar] = stdLayer.stdConv(net, outputVar, ...
    96, 'Conv5', 3, 64, 'poolingStride', 1, ...
    'pad', 1, 'batchNorm', opts.batchNorm);
% fc1
[net, outputVar] = stdLayer.stdDense(net, outputVar, 6, 64, 'Fc1', 384);
% fc2
[net, outputVar] = stdLayer.stdDense(net, outputVar, 1, 384, 'Fc2', 384);
% softmax classifier
[net, ~] = stdLayer.stdLoss(net, outputVar, 384, 'Cilia', 2);

% Meta parameters
net.meta.normalization.averageImage = [];
net.meta.trainOpts.learningRate = logspace(-3, -5, 60);
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = 256 ;

% Init paramters randomly
net.initParams();
end