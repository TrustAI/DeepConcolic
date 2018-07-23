%% get the box contraint
% please cite paper: Y. Sun, M. Wu, W. Ruan, X. Huang, M. Kwiatkowska and ...
% D. Kroening, Concolic Testing for Deep Neural Networks, 
% The 33rd IEEE/ACM International Conference on Automated Software Engineering (ASE'18),
% Montpellier, France, September 3 - 7, 2018. 
% Contact Author: Wenjie Ruan (wenjie.ruan@cs.ox.ac.uk)

function bounds = BoxBoundsConst(X0,d_bound,pixelRange)

X0_min = X0 - d_bound;
X0_max = X0 + d_bound;
pixelRange = repmat(pixelRange,size(X0));

bounds  = [max([pixelRange(:,1),X0_min],[],2),min([pixelRange(:,2),X0_max],[],2)];

end

