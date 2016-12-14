
% SoRec v0.9
% Modified using SGD
clear
clc
load epinion

 
% paramter initialization
% step = 0.001;
NUM_USER = 22166;
NUM_ITEM = 296277;


lambdaU  = 0.01 % Regularization parameter  
lambdaP = 0.01
lambdaC = 0.1;
featureDim = 8
testIdx = 1
maxIterNum = 50;
numBatch = 10; % number of batches

% split training set and test set
% sigmoid function is unused, if doesn't work. change that
gamma = 0.8; % coefficient for velocity
alpha = 0.001; % coefficient for gradient
trainMatrix = [];
socialMatrix = trustnetwork;
testMatrix = [];
for i = 1:5
    if i == testIdx
        testMatrix = rating{i};
    else
        trainMatrix = [trainMatrix;rating{i}];
    end
end

allData = [trainMatrix;testMatrix];
[numData,~] = size([allData]);

trainMatrix = trainMatrix(:,1:3);
testMatrix = testMatrix(:,1:3);
allData = allData(:,1:3);
[numTrain,~] = size(trainMatrix);
[numTest,~] = size(testMatrix);
[numSocial,~] = size(socialMatrix);

uvbatchSize = floor(numTrain/numBatch);
uzbatchSize = floor(numSocial/numBatch);


% feature initializtion
P = 0.1*randn(NUM_ITEM, featureDim); % Movie feature vectors
U = 0.1*randn(NUM_USER, featureDim); % User feature vecators
Z = 0.1*randn(NUM_USER,featureDim); % social feature vectors

dP = zeros(NUM_ITEM, featureDim); % gradients
dU = zeros(NUM_USER, featureDim); 
dZ = zeros(NUM_USER, featureDim); 

vU = zeros(NUM_USER, featureDim); % velocity
vP = zeros(NUM_ITEM, featureDim);
vZ = zeros(NUM_USER, featureDim);

socialMatrix(:,3) = socialMatrix(:,3)*5;

socialMean = mean(socialMatrix(:,3));
ratingMean = mean(trainMatrix(:,3));
trainMatrix(:,3) = trainMatrix(:,3) - ratingMean;
socialMatrix(:,3) = socialMatrix(:,3) - socialMean;

RMSEList = [];
MAEList = [];

% SGD iteration
for currentIter = 1:maxIterNum
    % update U and V
    updateUVIdx = randperm(numTrain);
    trainMatrix = trainMatrix(updateUVIdx,:);
    
    for batch = 1:numBatch
    
        userSampleID = double(trainMatrix(uvbatchSize*(batch-1)+1:uvbatchSize*batch,1));
        movieSampleID = double(trainMatrix(uvbatchSize*(batch-1)+1:uvbatchSize*batch,2));
        ratingSample = double(trainMatrix(uvbatchSize*(batch-1)+1:uvbatchSize*batch,3));

        prediction = sum(P(movieSampleID,:).* U(userSampleID,:),2);
        error = repmat(2*(prediction - ratingSample),1,featureDim);
        % update information for each non-zero entry
        updateU = error.*P(movieSampleID,:) + lambdaU*U(userSampleID,:);
        updateP = error.*U(userSampleID,:) + lambdaP*P(movieSampleID,:);
        dU = zeros(NUM_USER,featureDim);
        dP = zeros(NUM_ITEM,featureDim);
        % summing up gradients
        for idx = 1:uvbatchSize
            dU(userSampleID(idx),:) =  dU(userSampleID(idx),:) + updateU(idx,:);
            dP(movieSampleID(idx),:) =  dP(movieSampleID(idx),:) + updateP(idx,:);
        end        
        vU = gamma.*vU + alpha.*dU;
        vP = gamma.*vP + alpha.*dP;
        % update feature matrix
        U = U - vU;
        P = P - vP;        
    end
    
    % update U and Z
    updateUZIdx = randperm(numSocial);
    socialMatrix = socialMatrix(updateUZIdx,:);
    for batch = 1:numBatch
        user1SocialID = double(socialMatrix(uzbatchSize*(batch-1)+1:uzbatchSize*batch,1));
        user2SocialID = double(socialMatrix(uzbatchSize*(batch-1)+1:uzbatchSize*batch,2));
        trustSample =  double(socialMatrix(uzbatchSize*(batch-1)+1:uzbatchSize*batch,3));
        % estimate error
        trustPrediction = sum(U(user1SocialID,:).* Z(user2SocialID,:),2);
        trustError = repmat(2*(trustPrediction - trustSample),1,featureDim);
        
        % updateinformation for U and Z 
        updateU = trustError.*Z(user2SocialID,:) + lambdaC*U(user1SocialID,:);
        updateZ = trustError.*U(user1SocialID,:) + lambdaC*Z(user2SocialID,:);
        dU = zeros(NUM_USER,featureDim);
        dZ = zeros(NUM_USER,featureDim);
        for idx = 1:uzbatchSize
            dU(user1SocialID(idx),:) =  dU(user1SocialID(idx),:) + updateU(idx,:);
            dZ(user2SocialID(idx),:) =  dZ(user2SocialID(idx),:) + updateZ(idx,:);
        end
        vU = gamma.*vU + alpha.*dU;
        vZ = gamma.*vZ + alpha.*dZ;
        % update
        U = U - vU;
        Z = Z - vZ;
    end
        
    % calculate training error
    prediction = sum(P(trainMatrix(:,2),:).*U(trainMatrix(:,1),:),2);
%     objFunction = sum(prediction - ratingSample).^2 + 0.5*lambdaU*(norm(U,'fro').^2) + 0.5*lambdaP*((norm(P,'fro')).^2); 
    trainError = sqrt(sum((prediction - trainMatrix(:,3)).^2,1)./numTrain); 
    
    % calculate test error
    userTestID = double(testMatrix(:,1));
    movieTestID = double(testMatrix(:,2));
    ratingTest = double(testMatrix(:,3));
    testPrediction = sum(P(movieTestID,:).*U(userTestID,:),2)+ratingMean;

    testPrediction(find(testPrediction>5)) = 5; 
    testPrediction(find(testPrediction<1)) = 1; 
  
    testRMSE = sqrt(sum((testPrediction - ratingTest).^2)/numTest);
    testMAE = sum(abs(testPrediction - ratingTest))./numTest;
    RMSEList = [RMSEList,testRMSE];
    MAEList = [MAEList,testMAE];
    fprintf('the %1.1f iteration finished, training RMSE = %6.4f ,test RMSE = %6.4f, test MAE = %6.4f  \n', ...
              currentIter, trainError,testRMSE,testMAE);  
end 



