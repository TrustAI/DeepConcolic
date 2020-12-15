%% Normalized into 0 to 1
% please cite paper: Y. Sun, M. Wu, W. Ruan, X. Huang, M. Kwiatkowska and ...
% D. Kroening, Concolic Testing for Deep Neural Networks, 
% The 33rd IEEE/ACM International Conference on Automated Software Engineering (ASE'18),
% Montpellier, France, September 3 - 7, 2018. 
% Contact Author: Wenjie Ruan (wenjie.ruan@cs.ox.ac.uk)

function norm_out = norm0to1(cdata1)   
cdata1 = cdata1 - min(cdata1(:));
    norm_out = cdata1 ./ max(cdata1(:));