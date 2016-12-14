
% Base PMF
% Modified using SGD
clear
clc
load movielensFull
 
% paramter initialization
% step = 0.001;
NUM_USER = 6040;
NUM_MOVIE = 3952;



lambdaU  = 0.001 % Regularization parameter  
lambdaP = 0.001
featureDim = 3
testIdx = 3
maxIterNum = 200
numBatch = 10; % number of batches
% split training set and test set
gamma = 0.8; % coefficient for velocity
alpha = 0.001; % coefficient for gradient
trainMatrix = [];
testMatrix = [];
for i = 1:5
    if i == testIdx
        testMatrix = movieLens{i};
    else
        trainMatrix = [trainMatrix;movieLens{i}];
    end
end

allData = [trainMatrix;testMatrix];
[numData,~] = size([allData]);

trainMatrix = trainMatrix(:,1:3);
testMatrix = testMatrix(:,1:3);
allData = allData(:,1:3);
[numTrain,~] = size(trainMatrix);
[numTest,~] = size(testMatrix);

batchSize = floor(numTrain/numBatch);



% feature initializtion
P = 0.1*randn(NUM_MOVIE, featureDim); % Movie feature vectors
U = 0.1*randn(NUM_USER, featureDim); % User feature vecators

dP = zeros(NUM_MOVIE, featureDim); % gradients
dU = zeros(NUM_USER, featureDim); 
vU = zeros(NUM_USER, featureDim); % velocity
vP = zeros(NUM_MOVIE, featureDim);

ratingMean = mean(trainMatrix(:,3));
trainMatrix(:,3) = trainMatrix(:,3) - ratingMean;
RMSEList = [];
MAEList = [];

% SGD iteration
for currentIter = 1:maxIterNum
    updateIdx = randperm(numTrain);
    for batch = 1:numBatch
    
        userSampleID = double(trainMatrix(batchSize*(batch-1)+1:batchSize*batch,1));
        movieSampleID = double(trainMatrix(batchSize*(batch-1)+1:batchSize*batch,2));
        ratingSample = double(trainMatrix(batchSize*(batch-1)+1:batchSize*batch,3));

        prediction = sum(P(movieSampleID,:).* U(userSampleID,:),2);
        error = repmat(2*(prediction - ratingSample),1,featureDim);
        % update information for each non-zero entry
        updateU = error.*P(movieSampleID,:) + lambdaU*U(userSampleID,:);
        updateP = error.*U(userSampleID,:) + lambdaP*P(movieSampleID,:);
        dU = zeros(NUM_USER,featureDim);
        dP = zeros(NUM_MOVIE,featureDim);
        % summing up gradients
        for idx = 1:batchSize
            dU(userSampleID(idx),:) =  dU(userSampleID(idx),:) + updateU(idx,:);
            dP(movieSampleID(idx),:) =  dP(movieSampleID(idx),:) + updateP(idx,:);
        end
        
        vU = gamma.*vU + alpha.*dU;
        vP = gamma.*vP + alpha.*dP;
        % update feature matrix
        U = U - vU;
        P = P - vP;
        
%         fprintf('the %1.0f th batch has finished \n', batch);
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



