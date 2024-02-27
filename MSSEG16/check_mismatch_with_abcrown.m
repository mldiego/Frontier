%% Checking inconsistencies

% The NNV results and counterexamples for SAT cases were very helpful for us. 
% We have resolved some issues with alpha-beta-crown verifying segmented models
% but some discrepancies (NNV solving SAT but abcrown solving UNSAT) led us to making more fundamental checks.
% 
% Attached is a minimal python script (including package versions we checked) 
% examining a case where NNV produces a counterexample:
% MSSEG16/results/img_297_sliceSize_64_linf_pixels_10_eps_0.0001_region.txt
% 
% The model MSSEG16/onnx/model64.onnx is loaded, the input for the counterexample
% above is loaded (xce) and fed into the model in order to check the output (y) 
% against the output in the counterexample (yce). The two do not match, 
% which means we are effectively verifying different models (leading to abcrown's 
% determination of UNSAT in this case) and cannot compare results. 
% We have tried a couple things to fix the discrepancy: using Matlab's Fortran-style 
% reshaping/flattening doesn't qualitatively change the output much, 
% and loading the pytorch state dictionaries you supplied in the repo results 
% in a mismatch error with the onnx model so we can't check that. 
% If you supplied the model definitions corresponding to those state 
% dictionaries somewhere in the repo, I haven't seen it, but that would let us check it.

%% Python code
% import pandas as pd  # 1.3.4, 2.2.0                                                                                               
% import numpy as np  # 1.21.6, 1.26.3                                                                                              
% import onnx2torch  # 1.5.13                                                                                                       
% # import onnx2pytorch  # git+https://github.com/KaidiXu/onnx2pytorch@102cf22e64ea7fae9462c1ba0feaa250ac0bc628                     
% import torch  # 1.12.1, 2.1.2                                                                                                     
% import onnx  # 1.11.0, 1.15.0                                                                                                     
% 
% 
% modelname = "MSSEG16/onnx/model64.onnx"
% cename = "MSSEG16/results/img_297_sliceSize_64_linf_pixels_10_eps_0.0001_region.txt"
% 
% modelonnx = onnx.load(modelname)
% model = onnx2torch.convert(modelonnx)
% # model = onnx2pytorch.ConvertModel(modelonnx, experimental=True)                                                                 
% 
% # the following pytorch state dictionary appears incompatible with the model above                                                
% # --> produces RuntimeError: Error(s) in loading state_dict for GraphModule                                                       
% # sd = torch.load('MSSEG16/pytorch/model64.pth')                                                                                  
% # if ('state_dict' in sd):                                                                                                        
% #     sd = sd['state_dict']                                                                                                       
% # model.load_state_dict(sd)                                                                                                       
% 
% df = pd.read_table(cename)
% xce = []
% yce = []
% for field in df.values:
%     if ('X_' in field[0]):
%         xce.append(float(field[0].split()[1].replace(")", "")))
% 
%     if ('Y_' in field[0]):
%         yce.append(float(field[0].split()[1].replace(")", "")))
% 
% assert len(xce) == 4096
% assert len(yce) == 4096
% 
% xce = np.array(xce)
% yce = np.array(yce)
% ymodel = model(torch.Tensor(xce.reshape(1, 1, 64, 64)))
% y = ymodel.detach().numpy().flatten()
% 
% # MSSEG16/vnnlib/img_297_sliceSize_64_linf_pixels_10_eps_0.0001_region.vnnlib                                                     
% print("C order")
% print("Y_562 ", y[562], ' vs ', yce[562])
% print("Y_563 ", y[563], ' vs ', yce[563])
% print("Y_819 ", y[819], ' vs ', yce[819])
% # Y_562  0.8250625  vs  -0.1551035940647125                                                                                       
% # Y_563  0.9242837  vs  -0.293496310710907                                                                                        
% # Y_819  0.9987863  vs  -0.1542764157056808                                                                                       
% 
% np.savetxt('ximg.dat', xce.reshape(64, 64))
% np.savetxt('yimg.dat', ymodel.detach().numpy().squeeze())
% np.savetxt('yceimg.dat', yce.reshape(64, 64))
% np.savetxt('yflatten.dat', y, fmt="%g")
% 
% # Matlab-style column-major order                                                                                                 
% ymodel = model(torch.Tensor(xce.reshape(1, 1, 64, 64, order='F')))
% y = ymodel.detach().numpy().flatten(order='F')
% 
% print("F order")
% print("Y_562 ", y[562], ' vs ', yce[562])
% print("Y_563 ", y[563], ' vs ', yce[563])
% print("Y_819 ", y[819], ' vs ', yce[819])
% # Y_562  0.6908672  vs  -0.1551035940647125                                                                                       
% # Y_563  0.28768453  vs  -0.293496310710907                                                                                       
% # Y_819  -0.11402262  vs  -0.1542764157056808                                                                                     
% 
% np.savetxt('ximgF.dat', xce.reshape(64, 64, order='F'))
% np.savetxt('yimgF.dat', ymodel.detach().numpy().squeeze())
% np.savetxt('yceimgF.dat', yce.reshape(64, 64, order='F'))
% np.savetxt('yflattenF.dat', y, fmt="%g")

%% What about MATLAB?

cename = "results/img_297_sliceSize_64_linf_pixels_10_eps_0.0001_region.txt";

[x,y] = getCounterExample(cename);

model = importONNXNetwork("onnx\model64.onnx");
inputSize = model.Layers(1).InputSize;

x = reshape(x, inputSize);
% y = reshape(y, inputSize);

yF = predict(model,x);
yP = reshape(yF, [numel(yF) 1]);

disp(string(y(563)) +" , " + string(yP(563)) + "  ... " + "0.6908672")
disp(string(y(564)) +" , " + string(yP(564)) + "  ... " + "0.28768453")
disp(string(y(820)) +" , " + string(yP(820)) + " ... " + "-0.11402262")
disp(" ")

yP = yF';
yP = reshape(yP, [numel(yP) 1]);

disp(string(y(563)) +" , " + string(yP(563)) + "  ... " + "0.6908672")
disp(string(y(564)) +" , " + string(yP(564)) + "  ... " + "0.28768453")
disp(string(y(820)) +" , " + string(yP(820)) + " ... " + "-0.11402262")

disp(" ");
disp("............................................")
disp(" ");

x = reshape(x, inputSize);
% y = reshape(y, inputSize);

yF = predict(model,x');
yP = reshape(yF, [numel(yF) 1]);

disp(string(y(563)) +" , " + string(yP(563)) + "  ... " + "0.6908672")
disp(string(y(564)) +" , " + string(yP(564)) + "  ... " + "0.28768453")
disp(string(y(820)) +" , " + string(yP(820)) + " ... " + "-0.11402262")
disp(" ")

yP = yF';
yP = reshape(yP, [numel(yP) 1]);

disp(string(y(563)) +" , " + string(yP(563)) + "  ... " + "0.6908672")
disp(string(y(564)) +" , " + string(yP(564)) + "  ... " + "0.28768453")
disp(string(y(820)) +" , " + string(yP(820)) + " ... " + "-0.11402262")

% Notes
% Nothing seems to match the inference values from pytorch...
LayerType = [];
LayerName = [];
LayerWeights = {};  % Convolutional layers
LayerBias = {};     % convolutional layers
LayerScale = {};    % batchnorm
LayerOffset = {};   % batchnorm
LayerMean = {};     % batchnorm
LayerVariance = {}; % batchnorm
LayerOutput = {};   % all layers

% Get model data and send it their way
for i=1:length(model.Layers)
    LayerType = [LayerType; string(class(model.Layers(i)))];
    LayerName = [LayerName; string(model.Layers(i).Name)];
    LayerOutput{i} = activations(model, x, model.Layers(i).Name);
    if contains(class(model.Layers(i)), "Convolution")
        LayerWeights{i} = model.Layers(i).Weights;
        LayerBias{i} = model.Layers(i).Bias;
    elseif contains(class(model.Layers(i)), "BatchNormalization")
        LayerScale{i} = model.Layers(i).Scale;
        LayerOffset{i} = model.Layers(i).Offset;
        LayerMean{i} = model.Layers(i).TrainedMean;
        LayerVariance = model.Layers(i).TrainedVariance;
    end
end

save("model_info_for_mismatch.mat", "LayerBias", "LayerWeights", "LayerOutput",...
    "LayerName", "LayerType", "LayerVariance", "LayerMean", "LayerOffset", "LayerScale");

%% Helper functions
function [x,y] = getCounterExample(cename)
    fid = fopen(cename, "r");
    fgetl(fid); 
    fgetl(fid);
    fgetl(fid);
    % First three lines do nothing
    x = [];
    y = [];
    tline = fgetl(fid);
    while tline ~= ")" % end of file
        a = split(tline, " ");
        a = a{end}; 
        a = split(a, ")");
        a = a{1};
        a = str2double(a);
        if contains(tline, "X")
            x = [x ; a];
        else
            y = [y ; a];
        end
        tline = fgetl(fid);
    end

end