%% calculate the objective function
% please cite paper: Y. Sun, M. Wu, W. Ruan, X. Huang, M. Kwiatkowska and ...
% D. Kroening, Concolic Testing for Deep Neural Networks, 
% The 33rd IEEE/ACM International Conference on Automated Software Engineering (ASE'18),
% Montpellier, France, September 3 - 7, 2018. 
% Contact Author: Wenjie Ruan (wenjie.ruan@cs.ox.ac.uk)

function value = ObjectFunc1(x)

global layer
global convnet
global fVal_x0
global fInd_x0
global x0Vec
global delta
global resultCell
global cell_ind
% global lipConst
global value_Lip

% global lipConst
cell_ind = cell_ind + 1;

rowSize = length(x)^0.5;
X_image = reshape(x,[rowSize,rowSize,1]);
fVal_all = activations(convnet,X_image,layer,'OutputAs','rows');
[~,fVal_all_ind] = max(fVal_all);

fVal_j = fVal_all(:,fInd_x0);
value_Lip = abs(fVal_x0 - fVal_j)./(norm(x - x0Vec)+delta);
value = -abs(fVal_x0 - fVal_j); %+ lipConst*(norm(x - x0Vec));
%-value_Lip;

resultCell{cell_ind,1} = X_image;
resultCell{cell_ind,2} = double(value_Lip);
resultCell{cell_ind,3} = fVal_all_ind;
resultCell{cell_ind,4} = -double(value);


