%% add stop conditions
% please cite paper: Y. Sun, M. Wu, W. Ruan, X. Huang, M. Kwiatkowska and ...
% D. Kroening, Concolic Testing for Deep Neural Networks, 
% The 33rd IEEE/ACM International Conference on Automated Software Engineering (ASE'18),
% Montpellier, France, September 3 - 7, 2018. 
% Contact Author: Wenjie Ruan (wenjie.ruan@cs.ox.ac.uk)

function [stop,options,optchanged] = StopFunc(~,options,flag) %#ok<INUSD>
% global delta
global value_Lip
global lipConst

stop = false;
optchanged = false;
% value_Lip
if value_Lip > lipConst
    stop = true;
end

end

