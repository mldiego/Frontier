%% Create latex table with the vnnlib verification results

resultsnnv = dir('results/*.txt');

% begin table
fid = fopen("tableSummary.tex", "w");
fprintf(fid, "\\scriptsize\n");
fprintf(fid, "\\begin{longtable}{| l | c | c | c |}\n");
fprintf(fid, "\\toprule");
fprintf(fid, "\\textbf{Property} & \\textbf{Result} & \\textbf{V. Time} & \\textbf{C.S. Time} \\");
fprintf(fid, "\\");
fprintf(fid, "\n");
fprintf(fid, "\\midrule\n");
% process all results
for i=1:height(resultsnnv)
    name = resultsnnv(i).name;
    name = split(name, '_');
    name = strjoin(name, '\\_');
    name = replace(name, ".txt", "");
    [res, vt, ft] = processResultFile("results/"+resultsnnv(i).name);
    fprintf(fid, "$%s$ & %s & %f & %f \\", name, res, vt, ft);
    fprintf(fid, "\\");
    fprintf(fid, "\n");
end
fprintf(fid, "\\bottomrule\n"); 
fprintf(fid,"\\end{longtable}\n");
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