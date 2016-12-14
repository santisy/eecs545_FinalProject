
% Base PMF
clear
clc
load movielensFull
 
% paramter initialization
step = 0.001;
lambdaU  = 0.001; % Regularization parameter  
lambdaP = 0.001;
featureDim = 10;
testIdx = 1;
maxIterNum = 2000;

% split training set and test set
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
[numData,~] = size(allData);
% the former three columns to do the process
trainMatrix = trainMatrix(:,1:3);
testMatrix = testMatrix(:,1:3);
allData = allData(:,1:3);
[numTrainSample,~] = size(trainMatrix);
[numTestSample,~] = size(testMatrix);

numUser = 6040;
numMovie = 3952;

% feature initializtion
P = 0.1*randn(numMovie, featureDim); % Movie feature vectors
U = 0.1*randn(numUser, featureDim); % User feature vecators

dP = zeros(numMovie, featureDim); % gradients
dU = zeros(numUser, featureDim); 

ratingMean = mean(trainMatrix(:,3));
trainMatrix(:,3) = trainMatrix(:,3) - ratingMean;
RMSEList = [];
MAEList = [];
% Gradient descent iteration
for currentIter = 1:maxIterNum
    userSampleID = double(trainMatrix(:,1));
    movieSampleID = double(trainMatrix(:,2));
    ratingSample = double(trainMatrix(:,3));

    prediction = sum(P(movieSampleID,:).* U(userSampleID,:),2);
    error = repmat(2*(prediction - ratingSample),1,featureDim);
    % update information for each non-zero entry
    updateU = error.*P(movieSampleID,:) + lambdaU*U(userSampleID,:);
    updateP = error.*U(userSampleID,:) + lambdaP*P(movieSampleID,:);
    dU = zeros(numUser,featureDim);
    dP = zeros(numMovie,featureDim);
    % summing up gradients
    for idx = 1:numTrainSample
        dU(userSampleID(idx),:) =  dU(userSampleID(idx),:) + updateU(idx,:);
        dP(movieSampleID(idx),:) =  dP(movieSampleID(idx),:) + updateP(idx,:);
    end

    % update feature matrix
    U = U - step.*dU;
    P = P - step.*dP;
    
    % calculate training error
    prediction = sum(P(movieSampleID,:).*U(userSampleID,:),2);
%     objFunction = sum(prediction - ratingSample).^2 + 0.5*lambdaU*(norm(U,'fro').^2) + 0.5*lambdaP*((norm(P,'fro')).^2); 
    trainError = sqrt(sum((prediction - ratingSample).^2,1)./numTrainSample); 
    
    % calculate test error
    userTestID = double(testMatrix(:,1));
    movieTestID = double(testMatrix(:,2));
    ratingTest = double(testMatrix(:,3));
    testPrediction = sum(P(movieTestID,:).*U(userTestID,:),2)+ratingMean;

    testPrediction(find(testPrediction>5)) = 5; 
    testPrediction(find(testPrediction<1)) = 1; 
  
    testRMSE = sqrt(sum((testPrediction - ratingTest).^2)/numTestSample);
    testMAE = sum(abs(testPrediction - ratingTest))./numTestSample;
    RMSEList = [RMSEList,testRMSE];
    MAEList = [MAEList,testMAE];
    fprintf('the %1.1f iteration finished, training RMSE = %6.4f ,test RMSE = %6.4f, test MAE = %6.4f  \n', ...
              currentIter, trainError,testRMSE,testMAE);  
end 



