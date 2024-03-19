%% Checking for mismatch

torchmodel = importNetworkFromONNX('ptmodel.onnx');
onnxmodel  = importNetworkFromONNX('onnx/model64.onnx');

% Create simple input (0s)
x0_32 = zeros(64,64,'single');
x0_64 = zeros(64,64,'double');

% Compute outputs
y0_32_torch = predict(torchmodel,x0_32);
y0_64_torch = predict(torchmodel,x0_64);

y0_32_onnx = predict(onnxmodel,x0_32);
y0_64_onnx = predict(onnxmodel,x0_64);


% Create simple input (1s)
x1_32 = ones(64,64,'single');
x1_64 = ones(64,64,'double');

% Compute outputs
y1_32_torch = predict(torchmodel,x1_32);
y1_64_torch = predict(torchmodel,x1_64);

y1_32_onnx = predict(onnxmodel,x1_32);
y1_64_onnx = predict(onnxmodel,x1_64);

clear torchmodel onnxmodel;

save("restart_mismatch_data.mat");