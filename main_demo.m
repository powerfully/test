close all; clear; clc;
addpath(genpath('./utils/'));
addpath(genpath('./codes/'));

result_URL = './results/';
if ~isdir(result_URL)
    mkdir(result_URL);  
end
db = {'coco_cnn'};   
% db = {'WIKI','MIRFLICKR0','NUSWIDE10'};   
% hashmethods = {'LSSH','CMFH','SCM_{orth}','SePH','DCH','SCRATCH','BATCH','DOCH','GCN'};
hashmethods = {'SePH', 'SCRATCH','BATCH','DOCH'};
% hashmethods = {'BATCH'};
loopnbits = [16,32,64,128];
% loopnbits = [16];


for dbi = 1     :length(db)
    db_name = db{dbi}; param.db_name = db_name;
    
    diary(['./results/conv_',db_name,'_result.txt']);
    diary on;
    
    %% load dataset
    load(['./datasets/',db_name,'.mat']);
    final_result_name = [result_URL 'final_' db_name '_result' '.mat'];
    
        if strcmp(db_name, 'MIRFLICKR0')
        X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
        R = randperm(size(L,1));
        sampleInds = R(2001:end);
        queryInds = R(1:2000);
        XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
        XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
        param.top_K = 13000;
        clear X Y L
        

    elseif strcmp(db_name, 'NUSWIDE10')
        X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
        R = randperm(size(L,1));
        %sampleInds = R(2001:end);
        sampleInds = R(2001:2000+25000);
        queryInds = R(1:2000);
        XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
        XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
        param.top_K = 25000;
        clear X Y L
    elseif strcmp(db_name, 'coco_cnn')
        XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
        XTest = I_te; YTest = T_te; LTest = L_te;
        param.top_K = 18000;
        
    elseif strcmp(db_name, 'WIKI')
        c = length(unique(L_te));   %����
        
        [n_I_tr, ~] = size(L_tr);
        L = zeros(c,n_I_tr);
        for i = 1:n_I_tr
            a = L_tr(i);
            L(a,i) = 1;
        end
        [n_T_te, ~] = size(L_te);
        TL = zeros(c,n_T_te);
        for i = 1:n_T_te
            a = L_te(i);
            TL(a,i) = 1;
        end
%         queryInds = R(1:2);
%         sampleInds = R(1:2);
        queryInds=1;sampleInds=1;
        XTrain = I_tr; YTrain=T_tr;LTrain=L';
        XTest= I_te; YTest = T_te; LTest = TL';
        param.top_K = 1000;
        elseif strcmp(db_name, 'IAPRTC-12')
%        XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
        inx = randperm(size(L_tr,1),size(L_tr,1));
        XTrain = I_tr(inx, :); YTrain = T_tr(inx, :); LTrain = L_tr(inx, :);
        XTest = I_te; YTest = T_te; LTest = L_te;
%         clear X Y L
    end
%     clear I_tr I_te L_tr L_te
    
    %% Kernel representation
    param.nXanchors = 5000; param.nYanchors = 5000;
    if 1
        anchor_idx = randsample(size(XTrain,1), param.nXanchors);
        XAnchors = XTrain(anchor_idx,:);
        anchor_idx = randsample(size(YTrain,1), param.nYanchors);
        YAnchors = YTrain(anchor_idx,:);
    else
        [~, XAnchors] = litekmeans(XTrain, param.nXanchors, 'MaxIter', 30);
        [~, YAnchors] = litekmeans(YTrain, param.nYanchors, 'MaxIter', 30);
    end
    
    [XKTrain,XKTest]=Kernel_Feature(XTrain,XTest,XAnchors);
    [YKTrain,YKTest]=Kernel_Feature(YTrain,YTest,YAnchors);

    
    %% Label Format
% %     if isvector(LTrain)
% %         LTrain = sparse(1:length(LTrain), double(LTrain), 1); LTrain = full(LTrain);
% %         LTest = sparse(1:length(LTest), double(LTest), 1); LTest = full(LTest);
% %     end
%     if strcmp(db_name, 'mirflickr25k')
% %        XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
%         inx = randperm(size(L_tr,1),size(L_tr,1));
%         XTrain = I_tr(inx, :); YTrain = T_tr(inx, :); LTrain = L_tr(inx, :);
%         XTest = I_te; YTest = T_te; LTest = L_te;
% %         clear X Y L
% 
%     elseif strcmp(db_name, 'wiki_data')
% %        XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
%         inx = randperm(size(L_tr,1),size(L_tr,1));
%         XTrain = I_tr(inx, :); YTrain = T_tr(inx, :); LTrain = L_tr(inx, :);
%         XTest = I_te; YTest = T_te; LTest = L_te;
% %         clear X Y L
% 
%    elseif strcmp(db_name, 'nusData')
% %        XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
%         inx = randperm(size(L_tr,1),20000);
%         XTrain = I_tr(inx, :); YTrain = T_tr(inx, :); LTrain = L_tr(inx, :);
%         XTest = I_te; YTest = T_te; LTest = L_te;
% %         clear X Y L
% 
%     elseif strcmp(db_name, 'IAPRTC-12')
% %        XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
%         inx = randperm(size(L_tr,1),size(L_tr,1));
%         XTrain = I_tr(inx, :); YTrain = T_tr(inx, :); LTrain = L_tr(inx, :);
%         XTest = I_te; YTest = T_te; LTest = L_te;
% %         clear X Y L
% 
%     elseif strcmp(db_name, 'mir_cnn')
% %        XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
%         inx = randperm(size(L_tr,1),size(L_tr,1));
%         XTrain = I_db(inx, :); YTrain = T_db(inx, :); LTrain = L_db(inx, :);
%         XTest = I_te; YTest = T_te; LTest = L_te;
% %         clear X Y L
% 
%     elseif strcmp(db_name, 'MIRFLICKR_deep')
%         R = randperm(size(L,1));
%         sampleInds = R(2001:end);
%         queryInds = R(1:2000);
%          XTrain = X(sampleInds,:); YTrain = Y(sampleInds,:); LTrain = L(sampleInds,:);
%         XTest = X(queryInds,:); YTest = Y(queryInds,:); LTest = L(queryInds,:);
% %         clear X Y L
% 
%     elseif strcmp(db_name, 'WIKI_deep')
%         XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
%         XTest = I_te; YTest = T_te; LTest = L_te;
% %         clear X Y L
%         
%     elseif strcmp(db_name, 'NUS21_deep')
%         R = randperm(size(L,1));
%         sampleInds = R(2001:end);
%         queryInds = R(1:2000);
%          XTrain = X(sampleInds,:); YTrain = Y(sampleInds,:); LTrain = L(sampleInds,:);
%         XTest = X(queryInds,:); YTest = Y(queryInds,:); LTest = L(queryInds,:);
% %         clear X Y L
%     end
% %     clear I_tr I_te L_tr L_te
%     
%     %% Kernel representation
%     [n, ~] = size(YTrain);
%     if strcmp(db_name, 'mirflickr25k')
%         n_anchors = 1000;
%         anchor_image = XTrain(randsample(n, n_anchors),:); 
%         anchor_text = YTrain(randsample(n, n_anchors),:);
%         XKTrain = RBF_fast(XTrain',anchor_image'); XKTest = RBF_fast(XTest',anchor_image'); 
%         YKTrain = RBF_fast(YTrain',anchor_text');  YKTest = RBF_fast(YTest',anchor_text'); 
%      
%    elseif strcmp(db_name, 'nusData')
%         n_anchors = 1000;
%         anchor_image = XTrain(randsample(n, n_anchors),:); 
%         anchor_text = YTrain(randsample(n, n_anchors),:);
%         XKTrain = RBF_fast(XTrain',anchor_image'); XKTest = RBF_fast(XTest',anchor_image'); 
%         YKTrain = RBF_fast(YTrain',anchor_text');  YKTest = RBF_fast(YTest',anchor_text'); 
%         
%     if strcmp(db_name, 'IAPRTC-12')
%         n_anchors = 2000;
%         anchor_image = XTrain(randsample(n, n_anchors),:); 
%         anchor_text = YTrain(randsample(n, n_anchors),:);
%         XKTrain = RBF_fast(XTrain',anchor_image'); XKTest = RBF_fast(XTest',anchor_image'); 
%         YKTrain = RBF_fast(YTrain',anchor_text');  YKTest = RBF_fast(YTest',anchor_text'); 
%     end
%         
%     elseif strcmp(db_name, 'wiki_data')
%         n_anchors = 1000;
%         anchor_image = XTrain(randsample(n, n_anchors),:); 
%         anchor_text = YTrain(randsample(n, n_anchors),:);
%         XKTrain = RBF_fast(XTrain',anchor_image'); XKTest = RBF_fast(XTest',anchor_image'); 
%         YKTrain = RBF_fast(YTrain',anchor_text');  YKTest = RBF_fast(YTest',anchor_text'); 
%         
%     elseif strcmp(db_name, 'MIRFLICKR_deep')
%         param.nXanchors = 5000; param.nYanchors = 5000;
%         if 1
%             anchor_idx = randsample(size(XTrain,1), param.nXanchors);
%             XAnchors = XTrain(anchor_idx,:);
%             anchor_idx = randsample(size(YTrain,1), param.nYanchors);
%             YAnchors = YTrain(anchor_idx,:);
%             XKTrain = RBF_fast(XTrain',XAnchors'); XKTest = RBF_fast(XTest',XAnchors'); 
%             YKTrain = RBF_fast(YTrain',YAnchors');  YKTest = RBF_fast(YTest',YAnchors'); 
%         end
%         
%      elseif strcmp(db_name, 'NUS21_deep')
%         param.nXanchors = 5000; param.nYanchors = 5000;
%         if 1
%             anchor_idx = randsample(size(XTrain,1), param.nXanchors);
%             XAnchors = XTrain(anchor_idx,:);
%             anchor_idx = randsample(size(YTrain,1), param.nYanchors);
%             YAnchors = YTrain(anchor_idx,:);
%             XKTrain = RBF_fast(XTrain',XAnchors'); XKTest = RBF_fast(XTest',XAnchors'); 
%             YKTrain = RBF_fast(YTrain',YAnchors');  YKTest = RBF_fast(YTest',YAnchors');
%         end
        
%     elseif strcmp(db_name, 'wiki_data')
%         param.nXanchors = 500; param.nYanchors = 500;
%         if 1
%             anchor_idx = randsample(size(XTrain,1), param.nXanchors);
%             XAnchors = XTrain(anchor_idx,:);
%             anchor_idx = randsample(size(YTrain,1), param.nYanchors);
%             YAnchors = YTrain(anchor_idx,:);
%         end
%     end
    
    
    %% Label Format
%     if isvector(LTrain)
%         LTrain = sparse(1:length(LTrain), double(LTrain), 1); LTrain = full(LTrain);
%         LTest = sparse(1:length(LTest), double(LTest), 1); LTest = full(LTest);
%     end
    
    
    %% Methods
    eva_info = cell(length(hashmethods),length(loopnbits));
    
    for ii =1:length(loopnbits)
        fprintf('======%s: start %d bits encoding======\n\n',db_name,loopnbits(ii));
        param.nbits = loopnbits(ii);
        
        for jj = 1:length(hashmethods)
            switch(hashmethods{jj})
                case 'ILRH'
                    fprintf('......%s start...... \n\n', 'ILRH');
                    BATCHparam = param;
                    BATCHparam.lambda3 = 1e4; BATCHparam.belta = 8; BATCHparam.maxIter = 50;BATCHparam.numclass = size(LTrain,2);
                    if strcmp(db_name, 'wiki_data')
                        BATCHparam.alpha = 1e0; BATCHparam.lambda1 = 1e-1; BATCHparam.lambda2 = 1e1;
                    elseif strcmp(db_name, 'mirflickr25k')
                        BATCHparam.alpha = 1e-1; BATCHparam.lambda1 = 1e-2; BATCHparam.lambda2 = 1e0;
                    elseif strcmp(db_name, 'nusData')
                        BATCHparam.alpha = 1e-1; BATCHparam.lambda1 = 1e-2; BATCHparam.lambda2 = 1e0;
                    elseif strcmp(db_name, 'IAPRTC-12')
                        BATCHparam.alpha = 1e-1; BATCHparam.lambda1 = 1e-2; BATCHparam.lambda2 = 1e0;
                    end
                    I_te = bsxfun(@minus, XTest, mean(XTrain, 1));     %693*128
                    I_tr = bsxfun(@minus, XTrain, mean(XTrain, 1));     %
                    T_te = bsxfun(@minus, YTest, mean(YTrain, 1));     %693*10
                    T_tr = bsxfun(@minus, YTrain, mean(YTrain, 1));  
                    %%
                   
%                     X1_t = XTrain';
%                     X_raw_t = normalize(X1_t);        % 数据标准�?                  
%                     X_raw = X_raw_t';						   % 由于tsne标准输入数据以行向量表示，因此先转置
%                     Y = tsne(X_raw);                  % 得到的矩阵为Nx2，N为N个样本，Y矩阵�?320x2
%                     v1 = gscatter(Y(:,1), Y(:,2),LTrain,'','^');% 若无label输入，则画出的图没有色彩区分
%                     
%                     %
%                     Y1_t = YTrain';
%                     Y_raw_t = normalize(Y1_t);        % 数据标准�?                    
%                     Y_raw = Y_raw_t';						   % 由于tsne标准输入数据以行向量表示，因此先转置
%                     Y2 = tsne(Y_raw);                  % 得到的矩阵为Nx2，N为N个样本，Y矩阵�?320x2
%                     v2 = gscatter(Y2(:,1), Y2(:,2),LTrain,'','v');% 若无label输入，则画出的图没有色彩区分
%%
                    eva_info_ = evaluate_ILRH(I_tr',T_tr',LTrain,I_te',T_te',LTest,BATCHparam);
                    
                 case 'CMFH'
                    fprintf('......%s start...... \n\n', 'CMFH');
                    I_te = bsxfun(@minus, XTest, mean(XTrain, 1));
                    I_tr = bsxfun(@minus, XTrain, mean(XTrain, 1));
                    T_te = bsxfun(@minus, YTest, mean(YTrain, 1));
                    T_tr = bsxfun(@minus, YTrain, mean(YTrain, 1));
                    BATCHparam = param;
                    BATCHparam.lambda = 0.5; BATCHparam.mu = 100; BATCHparam.gamma = 0.01;
                    BATCHparam.maxIter = 30;
                    eva_info_ = evaluate_CMFH(I_tr',T_tr',LTrain,I_te,T_te,LTest,BATCHparam);
                    result_name = ['./results/',db_name,'_','CMFH','_',num2str(loopnbits(ii)),'.mat'];
                    save(result_name, 'eva_info_');
                             
                case 'BATCH'
                    fprintf('......%s start...... \n\n', 'BATCH');
                    BATCHparam = param;
                    BATCHparam.eta1 = 0.05; BATCHparam.eta2 = 0.05; BATCHparam.eta0 = 0.9;
                    BATCHparam.omega = 0.01; BATCHparam.xi = 0.01; BATCHparam.max_iter = 6;
                    eva_info_ = evaluate_BATCH(XKTrain,YKTrain,LTrain,XKTest,YKTest,LTest,BATCHparam);
                    result_name = ['./results/',db_name,'_','BATCH','_',num2str(loopnbits(ii)),'.mat'];
                    save(result_name, 'eva_info_');
                    
                case 'GCN'
                    fprintf('......%s start...... \n\n', 'GCN');
                    BATCHparam = param;
                    BATCHparam.eta1 = 0.05; BATCHparam.eta2 = 0.05; BATCHparam.eta0 = 0.9;
                    BATCHparam.omega = 0.01; BATCHparam.xi = 0.01; BATCHparam.max_iter = 6;
                    BATCHparam.lambda1 = [0];%0.01, 0.1, 1, 10, 100
                    BATCHparam.lambda2 = [100];
                    BATCHparam.lambda3 = [0.01];
                    BATCHparam.lambda4 = [0.01];
                    BATCHparam.bits=loopnbits(ii);
                    eva_info_ = evaluate_GCN(XKTrain,YKTrain,LTrain,XKTest,YKTest,LTest,BATCHparam);
                    result_name = ['./results/',db_name,'_','GCN','_',num2str(loopnbits(ii)),'.mat'];
                    save(result_name, 'eva_info_');
               
               case 'DLFH'
                    fprintf('......%s start...... \n\n', 'DLFH');
                    BATCHparam = param;
                    BATCHparam.maxIter = 30;
                    BATCHparam.gamma = 1e-6;
                    BATCHparam.lambda = 8;
                    BATCHparam.num_samples = param.nbits;
                    eva_info_ = evaluate_DLFH(XKTrain,YKTrain,LTrain,XKTest,YKTest,LTest,BATCHparam);

                case 'RDMH'
                    fprintf('......%s start...... \n\n', 'RDMH');
                    BATCHparam = param;
                    BATCHparam.lambdaX = 0.5; %
                    BATCHparam.beide = 1e0; %      0
                    BATCHparam.lambda = 1e1;  %    1
                    BATCHparam.theta = 1e-6; %    -6
                    BATCHparam.yeta = 1e-3; %     -3
                    BATCHparam.iter1 = 20;

                    XTest = bsxfun(@minus, XTest, mean(XTrain, 1)); XTrain = bsxfun(@minus, XTrain, mean(XTrain, 1));
                    YTest = bsxfun(@minus, YTest, mean(YTrain, 1)); YTrain = bsxfun(@minus, YTrain, mean(YTrain, 1));

                    [XKKTrain,XKKTest]=Kernelize(XTrain,XTest,db_name); [YKKTrain,YKKTest]=Kernelize(YTrain,YTest,db_name);
                    XKKTest = bsxfun(@minus, XKKTest, mean(XKKTrain, 1)); XKKTrain = bsxfun(@minus, XKKTrain, mean(XKKTrain, 1));
                    YKKTest = bsxfun(@minus, YKKTest, mean(YKKTrain, 1)); YKKTrain = bsxfun(@minus, YKKTrain, mean(YKKTrain, 1));
                    
                    eva_info_ = evaluate_RDMH(XKKTrain,YKKTrain,LTrain,XKKTest,YKKTest,LTest,BATCHparam);
                    
               case 'ASCSH'
                    fprintf('......%s start...... \n\n', 'ASCSH');
                    BATCHparam = param;
                    BATCHparam.maxIter = 25;
                    BATCHparam.gamma = 1e-6;
                    BATCHparam.lambda = 8;
                    BATCHparam.num_samples = 2 * param.nbits;
                    if strcmp(db_name, 'mirflickr25k')
                        BATCHparam.lambda_c = 1e-3;
                        BATCHparam.alpha1 = 5e-2;
                        BATCHparam.alpha2 = BATCHparam.alpha1;
                        BATCHparam.mu = 1e-4;
                        BATCHparam.eta = 0.005;
                    elseif strcmp(db_name, 'IAPRTC-12')
                        BATCHparam.lambda_c = 1e-5;
                        BATCHparam.alpha1 = 5e-3;
                        BATCHparam.alpha2 = BATCHparam.alpha1;
                        BATCHparam.mu = 1e-6;
                        BATCHparam.eta = 1e-4;
                    elseif strcmp(db_name, 'nusData')
                        BATCHparam.lambda_c = 1e-3;
                        BATCHparam.alpha1 = 5e-3;
                        BATCHparam.alpha2 = BATCHparam.alpha1;
                        BATCHparam.mu = 1e-4;
                        BATCHparam.eta = 1e-2;
                        BATCHparam.sc = 5000;
                    end
                    eva_info_ = evaluate_ASCSH(XKTrain,YKTrain,LTrain,XKTest,YKTest,LTest,BATCHparam);
                    
                case 'SCM_{seq}'
                    fprintf('......%s start...... \n\n', 'SCM_{seq}');
                    BATCHparam = param;
                    BATCHparam.lambda = 1e-6;
                    BATCHparam.maxIter = 25;
                    XMean = mean(XTrain);
                    XTrain = bsxfun(@minus, XTrain, XMean);
                    XTest = bsxfun(@minus, XTest, XMean);

                    YMean = mean(YTrain);
                    YTrain = bsxfun(@minus, YTrain, YMean);

                    YTest = bsxfun(@minus, YTest, YMean);
                    eva_info_ = evaluate_SCM_seq(XTrain,YTrain,LTrain,XTest,YTest,LTest,BATCHparam);
                    result_name = ['./results/',db_name,'_','SCM_{seq}','_',num2str(loopnbits(ii)),'.mat'];
                    save(result_name, 'eva_info_');
                    
                 case 'SCM_{orth}'
                    fprintf('......%s start...... \n\n', 'SCM_{orth}');
                    BATCHparam = param;
                    BATCHparam.lambda = 1e-6;
                    BATCHparam.maxIter = 25;
                    XMean = mean(XTrain);
                    XTrain = bsxfun(@minus, XTrain, XMean);
                    XTest = bsxfun(@minus, XTest, XMean);

                    YMean = mean(YTrain);
                    YTrain = bsxfun(@minus, YTrain, YMean);

                    YTest = bsxfun(@minus, YTest, YMean);
                    eva_info_ = evaluate_SCM_orh(XTrain,YTrain,LTrain,XTest,YTest,LTest,BATCHparam);
                    result_name = ['./results/',db_name,'_','SCM_{orth}','_',num2str(loopnbits(ii)),'.mat'];
                    save(result_name, 'eva_info_');
                     
                 case 'FSH'
                    fprintf('......%s start...... \n\n', 'FSH');
                    BATCHparam = param;
                    BATCHparam.k = 10; BATCHparam.Nsamp = 50; BATCHparam.iter = 100;
                    BATCHparam.lambda = 300; BATCHparam.lam = 0.5; BATCHparam.cca = 0; BATCHparam.km = 1;
                    I_te = normalize1(XTest); I_tr = normalize1(XTrain);
                    T_te = normalize1(YTest); T_tr = normalize1(YTrain);
                    I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
                    I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
                    T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
                    T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));
                    eva_info_ = evaluate_FSH(I_tr,T_tr,LTrain,I_te,T_te,LTest,BATCHparam);
                    
                case 'SMFH'
                    fprintf('......%s start...... \n\n', 'SMFH');
                    BATCHparam = param;
                    BATCHparam.alpha = 0.5;BATCHparam.beta = 100;BATCHparam.gamma = 1;
                    BATCHparam.lambda = 0.01;BATCHparam.k_nn = 5; %5
%                     indx = randperm(size(LTrain,1),10000);  %nus
%                     XTrai = XTrain(indx, :); YTrai = YTrain(indx, :); LTrai = LTrain(indx, :);
%                     I_te = bsxfun(@minus, XTest, mean(XTrai, 1));
%                     I_tr = bsxfun(@minus, XTrai, mean(XTrai, 1));
%                     T_te = bsxfun(@minus, YTest, mean(YTrai, 1));
%                     T_tr = bsxfun(@minus, YTrai, mean(YTrai, 1));
%                     eva_info_ = evaluate_SMFH(I_tr,T_tr,LTrai,I_te,T_te,LTest,BATCHparam);

                    I_te = bsxfun(@minus, XTest, mean(XTrain, 1));
                    I_tr = bsxfun(@minus, XTrain, mean(XTrain, 1));
                    T_te = bsxfun(@minus, YTest, mean(YTrain, 1));
                    T_tr = bsxfun(@minus, YTrain, mean(YTrain, 1));
                    eva_info_ = evaluate_SMFH(I_tr,T_tr,LTrain,I_te,T_te,LTest,BATCHparam);
                    
                case 'MTFH'
                    fprintf('......%s start...... \n\n', 'MTFH');
                    BATCHparam = param;
                    BATCHparam.alpha = 0.5;
                    BATCHparam.beta = 0.1;
                    BATCHparam.anchorNum = 500;
%         
%                     XTrai2 = XTrain(1:10000, :); YTrai2 = YTrain(1:10000, :); LTrai2 = LTrain(1:10000, :);%nus
%                     I_te = bsxfun(@minus, XTest, mean(XTrai2, 1));
%                     I_tr = bsxfun(@minus, XTrai2, mean(XTrai2, 1));
%                     T_te = bsxfun(@minus, YTest, mean(YTrai2, 1));
%                     T_tr = bsxfun(@minus, YTrai2, mean(YTrai2, 1));
%                     eva_info_ = evaluate_MTFH(I_tr,T_tr,LTrai2,I_te,T_te,L_te,BATCHparam);
                    
                    I_te = bsxfun(@minus, XTest, mean(XTrain, 1));
                    I_tr = bsxfun(@minus, XTrain, mean(XTrain, 1));
                    T_te = bsxfun(@minus, YTest, mean(YTrain, 1));
                    T_tr = bsxfun(@minus, YTrain, mean(YTrain, 1));
                    eva_info_ = evaluate_MTFH(I_tr,T_tr,LTrain,I_te,T_te,L_te,BATCHparam);
                    
                case 'SePH'
                    fprintf('......%s start...... \n\n', 'SePH');
                    BATCHparam = param;
                    BATCHparam.gamma = 1e-1; BATCHparam.lambda = 8; BATCHparam.rou = 1e6;
                    BATCHparam.eta = 1e-3; BATCHparam.alpha = 1e0; BATCHparam.maxIter = 25;
                    BATCHparam.num_samples = 2 * param.nbits;
                    eva_info_ = evaluate_SePH(XKTrain,YKTrain,LTrain,XKTest,YKTest,LTest,BATCHparam);
                    result_name = ['./results/',db_name,'_','SePH','_',num2str(loopnbits(ii)),'.mat'];
                    save(result_name, 'eva_info_');
                    
                case 'SRLCH'
                    fprintf('......%s start...... \n\n', 'SRLCH');
                    
                     
                    BATCHparam = param;
                    if strcmp(db_name, 'mirflickr25k')
                        BATCHparam.maxItr = 50;
                    elseif strcmp(db_name, 'IAPRTC-12')
                       BATCHparam.maxItr = 5;
                    elseif strcmp(db_name, 'wiki_data')
                       BATCHparam.maxItr = 5;
                    elseif strcmp(db_name, 'nusData')
                        BATCHparam.maxItr = 30;
                    end
                    
                    BATCHparam.muv = 1e-1;
                    BATCHparam.mut = 1e-5;
                    BATCHparam.lam = 1e-1;
                    BATCHparam.alp = 1e-2;
                    BATCHparam.bet = 1e-1;
                  
                    eva_info_ = evaluate_SRLCH(XTrain,YTrain,LTrain,XTest,YTest,LTest,BATCHparam);
                    
                case 'SCRATCH'%anchors = 500
                    fprintf('......%s start...... \n\n', 'SCRATCH');
                    BATCHparam = param;
                    BATCHparam.lambdaX = 0.5;
                    BATCHparam.alpha = 500;
                    BATCHparam.Xmu = 1000;
                    BATCHparam.gamma = 5;
                    BATCHparam.iter = 20;
                    param.nXanchors = 1000; param.nYanchors = 1000;
                    if 1
                        anchor_idx = randsample(size(XTrain,1), param.nXanchors);
                        XAnchors = XTrain(anchor_idx,:);
                        anchor_idx = randsample(size(YTrain,1), param.nYanchors);
                        YAnchors = YTrain(anchor_idx,:);
                    end
                    [XKTr,XKTe]=Kernel_Feature(XTrain,XTest,XAnchors);
                    [YKTr,YKTe]=Kernel_Feature(YTrain,YTest,YAnchors);              
                    eva_info_ = evaluate_SCRATCH(XKTr,YKTr,LTrain,XKTe,YKTe,LTest,BATCHparam);
                    result_name = ['./results/',db_name,'_','SCRATCH','_',num2str(loopnbits(ii)),'.mat'];
                    save(result_name, 'eva_info_');
                    
                case 'LCMFH'
                    fprintf('......%s start...... \n\n', 'LCMFH');
                    BATCHparam = param;
                    BATCHparam.lambda = 0.5; BATCHparam.mu = 1e1; BATCHparam.gamma = 1e-3;%1 -3
                    BATCHparam.maxIter = 20;
                    I_te = bsxfun(@minus, XTest, mean(XTrain, 1));
                    I_tr = bsxfun(@minus, XTrain, mean(XTrain, 1));
                    T_te = bsxfun(@minus, YTest, mean(YTrain, 1));
                    T_tr = bsxfun(@minus, YTrain, mean(YTrain, 1));
                    eva_info_ = evaluate_LCMFH(I_tr',T_tr',LTrain,I_te,T_te,LTest,BATCHparam);
                
                case 'DCH'
                    fprintf('......%s start...... \n\n', 'DCH');
                    BATCHparam = param;
                    BATCHparam.sigma = 0.2; %0.4
                    BATCHparam.maxItr = 5;
                    BATCHparam.gmap.lambda = 1; 
                    BATCHparam.gmap.loss = 'L2';
                    BATCHparam.Fmap.type = 'RBF';
                    BATCHparam.Fmap.nu = 0; %  penalty parm for F term
                    BATCHparam.Fmap.mu = 0;
                    BATCHparam.Fmap.lambda = 1e-2;
                    BATCHparam.anchors = 1000;

                    I_te = bsxfun(@minus, XKTest, mean(XKTrain, 1));
                    I_tr = bsxfun(@minus, XKTrain, mean(XKTrain, 1));
                    T_te = bsxfun(@minus, YKTest, mean(YKTrain, 1));
                    T_tr = bsxfun(@minus, YKTrain, mean(YKTrain, 1));
                    eva_info_ = evaluate_DCH(I_tr,T_tr,LTrain,I_te,T_te,LTest,BATCHparam);
                    result_name = ['./results/',db_name,'_','DCH','_',num2str(loopnbits(ii)),'.mat'];
                    save(result_name, 'eva_info_');
                    
                case 'LFMH'
                    fprintf('......%s start...... \n\n', 'LFMH');
                    BATCHparam = param;
                    BATCHparam.alphas = [0.5 0.5]; BATCHparam.beide= 1e-1; BATCHparam.gamma = 1e0;
                    BATCHparam.lamda=  1e-2; BATCHparam.iterum = 50;
                    I_te = bsxfun(@minus, XTest, mean(XTrain, 1));
                    I_tr = bsxfun(@minus, XTrain, mean(XTrain, 1));
                    T_te = bsxfun(@minus, YTest, mean(YTrain, 1));
                    T_tr = bsxfun(@minus, YTrain, mean(YTrain, 1));
                    I_temp = I_tr';
                    T_temp = T_tr';
                    [row, col]= size(I_temp);
                    [rowt, colt] = size(T_temp);

                    I_temp = bsxfun(@minus,I_temp , mean(I_temp,2));
                    T_temp = bsxfun(@minus,T_temp, mean(T_temp,2));

                    L = normalizeFea(LTrain);
                    eva_info_ = evaluate_LFMH(I_temp,T_temp,LTrain',I_te,T_te,LTest,BATCHparam); 
                    
                case 'LSSH'
                    fprintf('......%s start...... \n\n', 'LSSH');
                    BATCHparam = param;
                    BATCHparam.bits = loopnbits(ii);
                    BATCHparam.mu = 0.05;
                    BATCHparam.rho = 0.5;
                    BATCHparam.lambda = 0.2;
                    BATCHparam.maxOutIter = 20;
                    
                    I_te = bsxfun(@minus, XTest, mean(I_tr, 1));
%                     I_db = bsxfun(@minus, I_db, mean(I_tr, 1));
                    I_tr = bsxfun(@minus, XTrain, mean(I_tr, 1));

                    T_te = bsxfun(@minus, YTest, mean(T_tr, 1));
%                     T_db = bsxfun(@minus, T_db, mean(T_tr, 1));
                    T_tr = bsxfun(@minus, YTrain, mean(T_tr, 1));
                    %     % construct training set
                    I_temp = I_tr';
                    T_temp = T_tr';
                    [row, col] = size(I_temp);
                    [rowt, colt] = size(T_temp);

                    I_temp = bsxfun(@minus, I_temp, mean(I_temp, 2));
                    T_temp = bsxfun(@minus, T_temp, mean(T_temp, 2));
                    Im_te = (bsxfun(@minus, I_te', mean(I_tr', 2)))';
                    Te_te = (bsxfun(@minus, T_te', mean(T_tr', 2)))';
                    
                    eva_info_ = evaluate_LSSH(I_temp,T_temp,LTrain,Im_te,Te_te,LTest,BATCHparam);   
                    result_name = ['./results/',db_name,'_','LSSH','_',num2str(loopnbits(ii)),'.mat'];
                    save(result_name, 'eva_info_');
                    
                case 'DOCH'
                    fprintf('......%s start...... \n\n', 'DOCH');
                    BATCHparam = param;
                    BATCHparam.alpha = 0.25;
                    BATCHparam.num_anchor = 50;
                    BATCHparam.iter = 3; 
                    BATCHparam.theta = 0.1;
                    BATCHparam.chunk_size = 2000;
%                     XKTrain = XKTrain(1:2000,:);
%                     YKTrain = YKTrain(1:2000,:);
%                     LTrain = LTrain(1:2000,:);
                    eva_info_ = evaluate_DOCH(XKTrain,YKTrain,LTrain,XKTest,YKTest,LTest,BATCHparam);
                    result_name = ['./results/',db_name,'_','DOCH','_',num2str(loopnbits(ii)),'.mat'];
                    save(result_name, 'eva_info_');
            end
            
            
            eva_info{jj,ii} = eva_info_;
            clear eva_info_
        end
    end
    
    
    %% Results
    for ii = 1:length(loopnbits)
        for jj = 1:length(hashmethods)
            % MAP
            Image_VS_Text_MAP{jj,ii} = eva_info{jj,ii}.Image_VS_Text_MAP;
            Text_VS_Image_MAP{jj,ii} = eva_info{jj,ii}.Text_VS_Image_MAP;

            % Precision VS Recall
            Image_VS_Text_recall{jj,ii,:}    = eva_info{jj,ii}.Image_VS_Text_recall';
            Image_VS_Text_precision{jj,ii,:} = eva_info{jj,ii}.Image_VS_Text_precision';
            Text_VS_Image_recall{jj,ii,:}    = eva_info{jj,ii}.Text_VS_Image_recall';
            Text_VS_Image_precision{jj,ii,:} = eva_info{jj,ii}.Text_VS_Image_precision';

            % Top number Precision
            Image_To_Text_Precision{jj,ii,:} = eva_info{jj,ii}.Image_To_Text_Precision;
            Text_To_Image_Precision{jj,ii,:} = eva_info{jj,ii}.Text_To_Image_Precision;
            
            % Time
            trainT{jj,ii} = eva_info{jj,ii}.trainT;
            testT{jj,ii} = eva_info{jj,ii}.compressT;
        end
    end

    save(final_result_name,'eva_info','BATCHparam','loopnbits','hashmethods',...
        'trainT','testT','Image_VS_Text_MAP','Text_VS_Image_MAP','Image_VS_Text_recall','Image_VS_Text_precision',...
        'Text_VS_Image_recall','Text_VS_Image_precision','Image_To_Text_Precision','Text_To_Image_Precision','-v7.3');
    
    diary off;
end
