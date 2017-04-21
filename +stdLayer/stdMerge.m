function [net, outputVar] = stdMerge(net, inputVars, outputName, varargin)
% a standard api to merge layers

% parse params
if nargin < 3, error('Not enough arguments.'); end
opts.dim = 3;
opts = vl_argparse(opts, varargin);

% create merge block
mergeLayer = dagnn.Concat();
mergeLayer.dim = opts.dim;

% add layer
outputVar = [outputName, 'X'];
net.addLayer(outputName, mergeLayer, inputVars, outputVar);

end