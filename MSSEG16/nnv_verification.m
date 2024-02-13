%% Verify all onnx/vnnlib combos using NNV

addpath(genpath("../../nnv/code/nnv/"));

if ~isfolder("results")
    mkdir("results")
end

% get list of neural networks and properties
% networks = dir('onnx/*.onnx');
specs = dir('vnnlib/*.vnnlib');

% Analyze all benchmarks
for i=1:height(specs)

    % Verification outline
    %  1) Load components
    %  2) SAT? - Randomly evaluate network to search for counterexample 
    %  3) UNSAT? 
    %     a) Compute reachability 
    %     b) Verify (compute intersection between reach set and output property halfspace)
    %  4) Save results (computation and reachability)

     %% 1) Load components

    % Get path to vnnlib specification
    specPath = fullfile(specs(i).folder, specs(i).name);
    sliceSize = split(specs(i).name, "_");
    sliceSize = sliceSize{4};

    % verify only those with sliceSize = 64 (for now)
    if ~strcmp(sliceSize, '64')
        continue
    end

     % Get path to neural network
    onnxPath = "onnx/model"+sliceSize+".onnx";
    % load NN from onnx
    net = importONNXNetwork(onnxPath);
    % transform to NNV
    nnvnet = matlab2nnv(net);
    % get input size to resize input vectors into that shape
    inputSize = net.Layers(1).InputSize; % slice size
   
    % Create path to utput file (to save results)
    outputfile = fullfile("results", specs(i).name);
    outputfile = replace(outputfile, ".vnnlib", ".txt");

    % Load property to verify
    property = load_vnnlib(specPath);
    lb = single(property.lb); % input lower bounds
    ub = single(property.ub); % input upper bounds
    prop = property.prop; % output spec to verify
    
    
    %% 2) SAT?
    % Can we find a counterexample that violates the specification?

    status = 2; % initialize verification status -> unknown to start

    t = tic; % start timer
    
    rng(0); % set radom seed fix (for reproducibility)
    nRand = 100; % number of random inputs (can increase/decrease this as desired
    
    % Choose how to falsify based on vnnlib file
    if ~isa(lb, "cell") && length(prop) == 1 % one input, one output 
        counterEx = search_for_counterexamples(net, lb, ub, nRand, prop{1}.Hg, inputSize);
    else
        status = -1;
        warning("Working on adding support to other vnnlib properties")
    end
    
    cTime = toc(t);
    
    %% 3) UNSAT?
    % Verify specification using reachability analysis with Star sets
    
    % Define reachability options
    reachOptions = struct;
    % reachOptions.reachMethod = 'approx-star';
    % reachOptions.lp_solver = "glpk";
    reachOptions.reachMethod = 'relax-star-range';
    reachOptions.relaxFactor = 0.95;
    
    % Check if property was violated earlier
    if iscell(counterEx)
        status = 0;
    end
    
    t = tic;

    if status == 2 && isa(nnvnet, "NN") % no counterexample found and supported for reachability (otherwise, skip step 3 and write results)
    
    % Choose how to verify based on vnnlib file
        if ~isa(lb, "cell") && length(prop) == 1 % one input, one output

            % Get input set
            if ~isscalar(inputSize)
                lb = reshape(lb, inputSize);
                ub = reshape(ub, inputSize);
            end
            IS = ImageStar(lb, ub);

            
            try
                % Compute reachability
                ySet = nnvnet.reach(IS, reachOptions);
                
                % Verify property
                status = verify_specification(ySet, prop);
            catch
                status = -1; % error
            end
    
        else % for code clarity other options have been cleared as all specs are the same
            status = -1;
            warning("Working on adding support to other vnnlib properties")
        end
    
    end
    

    %% 4) Process results

    vTime = toc(t); % save total computation time
    
    % Write results to output file
    if status == 0
        fid = fopen(outputfile, 'w');
        fprintf(fid, 'sat \n');
        fprintf(fid, 'Inference time = %f \n', cTime);
        fprintf(fid, 'Verification time = %f \n', vTime);
        fclose(fid);
        write_counterexample(outputfile, counterEx)
    elseif status == 1
        fid = fopen(outputfile, 'w');
        fprintf(fid, 'unsat \n');
        fprintf(fid, 'Inference time = %f \n', cTime);
        fprintf(fid, 'Verification time = %f \n', vTime);
        fclose(fid);
    elseif status == 2
        fid = fopen(outputfile, 'w');
        fprintf(fid, 'unknown \n');
        fprintf(fid, 'Inference time = %f \n', cTime);
        fprintf(fid, 'Verification time = %f \n', vTime);
        fclose(fid);
    elseif status == -1
        fid = fopen(outputfile, 'w');
        fprintf(fid, 'Something did not work with the vnnlib file');
        fclose(fid);
    end

end



%% Helper Functions

% create random examples given upper and lower bounds
function xRand = create_random_examples(net, lb, ub, nR, inputSize)
    % get box from input bounds 
    xB = Box(lb, ub); % lb, ub must be vectors
    % Generate samples
    xRand = xB.sample(nR-2);
    xRand = [lb, ub, xRand];
    % reshape vectors into net input size
    xRand = reshape(xRand,[inputSize nR]); 
    if isa(net, 'dlnetwork') % need to convert to dlarray
        xRand = dlarray(xRand, "SSCB");
    end
end

% Search for counerexamples within the given input set
function counterEx = search_for_counterexamples(net, lb, ub, nRand, Hs, inputSize)
    % initialize vars
    counterEx = nan; % if nan, no counterexamples
    xRand = create_random_examples(net, lb, ub, nRand, inputSize);
    s = size(xRand);
    n = length(s);

    %  look for counterexamples
    for i=1:s(n)
        % get input example from xRand (all contained in input set)
        x = get_example(xRand, i);
        % predict output
        yPred = predict(net, x);
        if isa(x, 'dlarray') % if net is a dlnetwork
            x = extractdata(x);
            yPred = extractdata(yPred);
        end
        
        % check if property violated
        yPred = reshape(yPred, [], 1); % convert to column vector (if needed)
        for h=1:length(Hs)
            if Hs(h).contains(double(yPred)) % property violated % could one of the problems arise when we convert the output to double? It should not ideally...
                counterEx = {x; yPred}; % save input/output of countex-example
                break;
            end
        end
    end
    
end

% get example from the list xRand (randomly generated)
function x = get_example(xRand,i)
    s = size(xRand);
    n = length(s);
    if n == 4
        x = xRand(:,:,:,i);
    elseif n == 3
        x = xRand(:,:,i);
    elseif n == 2
        x = xRand(:,i);
        xsize = size(x);
        if xsize(1) ~= 1 && ~isa(x,"dlarray")
            x = x';
        end
    else
        error("InputSize = "+string(s));
    end
end

% write counter example to output file
function write_counterexample(outputfile, counterEx)
    % First line - > sat
    % after that, write the variables for each input dimension  of the counterexample
    %
    % Example:
    %  ( (X_0 0.12132)
    %    (X_1 3.45454)
    %    ( .... )
    %    (Y_0 2.32342)
    %    (Y_1 3.24355)
    %    ( ... )
    %    (Y_N 0.02456))
    %

    precision = '%.16g'; % set the precision for all variables written to txt file
    % open file and start writing counterexamples
    fid = fopen(outputfile, 'a+');
    x = counterEx{1};
    x = reshape(x, [], 1);
    % begin specifying value for input example
    fprintf(fid,'(');
    for i = 1:length(x)
        fprintf(fid, "(X_" + string(i-1) + " " + num2str(x(i), precision)+ ")\n");
    end
    y = counterEx{2};
    y = reshape(y, [], 1);
    % specify values for output example
    for j =1:length(y)
        fprintf(fid, "(Y_" + string(j-1) + " " + num2str(y(j), precision)+ ")\n");
    end
    fprintf(fid, ')');
    % close and save file
    fclose(fid);

end

