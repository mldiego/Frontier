%% Create latex table with the vnnlib verification results

resultsnnv = dir('results/*.txt');

% begin table
fid = fopen("tableSummary.tex", "w");
fprintf(fid, "\\begin{center}\n");
fprintf(fid, "\\begin{tabular}{ c c c c}\n");
fprintf(fid, "\\textbf{Property} & \\textbf{Result} & \\textbf{Verification Time} & \\textbf{Counterexample Search Time} \\");
fprintf(fid, "\\");
fprintf(fid, "\n");
% process all results
for i=1:height(resultsnnv)
    name = resultsnnv(i).name;
    name = split(name, '_');
    name = strjoin(name, '\\_');
    [res, vt, ft] = processResultFile("results/"+resultsnnv(i).name);
    fprintf(fid, " $%s$ & %s & %f & %f \\", name, res, vt, ft);
    fprintf(fid, "\\");
    fprintf(fid, "\n");
end
 % cell1 & cell2 & cell3 \\ 
 % cell4 & cell5 & cell6 \\  
 % cell7 & cell8 & cell9    
fprintf(fid,"\\end{tabular}\n");
fprintf(fid,"\\end{center}");
fclose(fid);


%% Helper functions
function [res, vt, ft] = processResultFile(resFile)
    % open result file and read line by line
    rid = fopen(resFile, 'r');
    % 1) verification result
    res = fgetl(rid); 
    % 2) falsification time (counterexample search)
    ft = fgetl(rid);  
    ft = split(ft, "=");
    ft = str2double(ft{end});
    % 3) verification time
    vt = fgetl(rid);  
    vt = split(vt, "=");
    vt = str2double(vt{end});
    fclose(rid);      % close file
end