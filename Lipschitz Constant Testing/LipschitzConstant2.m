%% Lipschitz Constants for DNNs
clear;
clc;

%% set up global variables
global layer
global convnet
global fVal_x0
global fInd_x0
global x0Vec
global delta
global lipConst
global resultCell
global cell_ind
%% prepare data

load DNN7

convnet = convnet1;
layer = 'fc_2';
maxConcolicNum = 15;
allResult = {};
kk = 1;
%for lipConst = [8 11:14]
for imgInd = 34 : 1 : 48
    
    tStart = tic;
    epsilonf = 0.001;
    pixelRange = [0,1];
    delta = 0.0039*0.001;
    x0 = XTest(:,:,:,imgInd);
    x0Vec = x0(:);
    boxConSize = 0.1;
    resultCell = {};
    cell_ind = 1;
    fAll = activations(convnet,x0,layer);
    rowSize = size(x0,1);
    
    [fVal_x0,fInd_x0] = max(fAll);
    
    boxBound = BoxBoundsConst(x0Vec,boxConSize,pixelRange);
    
    x = x0Vec;
    
    resultCell{cell_ind,1} = x0;
    resultCell{cell_ind,2} = 0;
    resultCell{cell_ind,3} = fInd_x0;
    resultCell{cell_ind,4} = 0;
    
    %% Stage one
    
    %         options = optimoptions('patternsearch','Display','off',...
    %             'OutputFcn',@StopFunc);
    options = optimoptions('patternsearch','Display','off');
    options.MeshContractionFactor = 0.5;
    options.MeshExpansionFactor = 2;
    options.MeshTolerance = 0.0039/2;
    options.UseParallel = true;
    options.MaxIterations = 100;
    x_2 = x;
    [x_opt,fval_opt,exitflag,output] = patternsearch(@ObjectFunc,x,[],[],[],[],...
        boxBound(:,1),boxBound(:,2),options);
    x_1 = x_opt;
    tmep_fval = 0;
    concolicNum = 1;
    lipAll = cell2mat(resultCell(:,2));
    
    maxLip = max(lipAll);
    fprintf('Img_Index = %d, Concolic_Num = %d, f_val = %4.2f, Lipschitz = %4.2f\n',...
        imgInd, concolicNum, fval_opt,maxLip);
    %% Stage two
    while(abs(tmep_fval-fval_opt)>epsilonf && concolicNum < maxConcolicNum)
        tic
        tmep_fval = fval_opt;
        concolicNum = concolicNum + 1;
        X1_image = reshape(x_1,[rowSize,rowSize,1]);
        fAll = activations(convnet,x0,layer);
        fVal_x0 = fAll(:,fInd_x0);
        
        [x_opt,fval_opt,exitflag,output] = patternsearch(@ObjectFunc1,x_1,[],[],[],[],...
            boxBound(:,1),boxBound(:,2),options);
        x_2 = x_1;
        x_1 = x_opt;
        
        lipAll = cell2mat(resultCell(:,2));
        
        maxLip = max(lipAll);
        fprintf('Img_Index = %d, Concolic_Num = %d, f_val = %4.2f, Lipschitz = %4.2f\n',...
            imgInd, concolicNum, fval_opt,maxLip);
        toc
    end
    tElapsed = toc(tStart);

    
    allResult{kk,1} = imgInd;
    allResult{kk,2} = lipConst;
    allResult{kk,3} = reshape(x_1,[rowSize,rowSize]);
    allResult{kk,4} = reshape(x_2,[rowSize,rowSize]);
    allResult{kk,5} = resultCell{end,2};
    allResult{kk,6} = tElapsed;
    
    resultCell = resultCell(1:10:end,:);
    fileName = ['resultCell_img' num2str(imgInd) '_Lip' num2str(lipConst) '.mat'];
    save(fileName,'resultCell');
    
    kk = kk+1;
%     save LipschitzTestAll_20180425_3 allResult
end
%end

%%
% x_img = reshape(x,size(x0));
% figure;
% subplot(1,3,1)
% imshow(x0);
% subplot(1,3,2)
% imshow(abs(x0-x_img));
% subplot(1,3,3)
% imshow(x_img);


