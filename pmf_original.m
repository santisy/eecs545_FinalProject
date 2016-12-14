% Version 1.000
%
% Code provided by Ruslan Salakhutdinov
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

%???????????????????MATLAB???.mat?????????
clc,clear
close all
restart = 1;
rand('state',0); 
randn('state',0); 

if restart==1 
  restart=0;
  epsilon=40; % Learning rate 
  lambda  = 0.02; % Regularization parameter 
  momentum=0.4; 
  numbatches= 20;
  
  epoch=1; 
  maxepoch=70; 

  load moviedata % Triplets: {user_id, movie_id, rating} 
  mean_rating = mean(train_vec(:,3)); %???????
 
  pairs_tr = length(train_vec); % training data ;900000
  pairs_pr = length(probe_vec); % validation data ;100209

   % Number of batches  
  num_m = 3952;  % Number of movies 
  num_p = 6040;  % Number of users 
  num_feat = 10; % Rank 10 decomposition 

  w1_M1     = 0.1*randn(num_m, num_feat); % Movie feature vectors
  w1_P1     = 0.1*randn(num_p, num_feat); % User feature vecators
  w1_M1_inc = zeros(num_m, num_feat);
  w1_P1_inc = zeros(num_p, num_feat);
  
  aa_ma = train_vec(:,2);
  aa_pa = train_vec(:,1);
  ratings_a = train_vec(:,3);
  
  L_iter = 1;
end

for i = 1:L_iter
for epoch = epoch:maxepoch
  rr = randperm(pairs_tr);  
  train_vec_rand = train_vec(rr,:);%????????
  clear rr 

  for batch = 1:numbatches %1-9
    fprintf(1,'epoch %d batch %d \r',epoch,batch);
    N=pairs_tr/numbatches; % number training triplets per batch 

    % ?????90????9?batch???batch?10??
    aa_p   = double(train_vec_rand((batch-1)*N+1:batch*N,1)); 
    aa_m   = double(train_vec_rand((batch-1)*N+1:batch*N,2));
    rating = double(train_vec_rand((batch-1)*N+1:batch*N,3));

    rating = rating-mean_rating; % Default prediction is the mean rating. 

    %%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
    pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
    
    %?Probabilistic Matrix Factorization????4.
    f = sum( (pred_out - rating).^2 + ...
        0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));

    %%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
    IO = repmat(2*(pred_out - rating),1,num_feat);
    Ix_m=IO.*w1_P1(aa_p,:) + lambda*w1_M1(aa_m,:);
    Ix_p=IO.*w1_M1(aa_m,:) + lambda*w1_P1(aa_p,:);

    dw1_M1 = zeros(num_m,num_feat);
    dw1_P1 = zeros(num_p,num_feat);

    for ii=1:N
      dw1_M1(aa_m(ii),:) =  dw1_M1(aa_m(ii),:) +  Ix_m(ii,:);
      dw1_P1(aa_p(ii),:) =  dw1_P1(aa_p(ii),:) +  Ix_p(ii,:);
    end

    %%%% Update movie and user features %%%%%%%%%%%
    % ? http://mooc.guokr.com/note/9711/ ? Momentum????????????????????????????????????????
    %???????????????????????????
    w1_M1_inc = momentum*w1_M1_inc + epsilon*dw1_M1/N;
    w1_M1 =  w1_M1 - w1_M1_inc;

    w1_P1_inc = momentum*w1_P1_inc + epsilon*dw1_P1/N;
    w1_P1 =  w1_P1 - w1_P1_inc;
  end 

  %%%%%%%%%%%%%% Compute Predictions after Paramete Updates %%%%%%%%%%%%%%%%%
  pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
  f_s = sum( (pred_out - rating).^2 + ...
        0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));
  err_train(maxepoch*(i-1)+epoch) = sqrt(f_s/N);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 
  NN=pairs_pr; %???????

  aa_p = double(probe_vec(:,1));
  aa_m = double(probe_vec(:,2));
  rating = double(probe_vec(:,3));

  %????????????????
  pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2) + mean_rating;
  ff = find(pred_out>5); pred_out(ff)=5; % Clip predictions 
  ff = find(pred_out<1); pred_out(ff)=1;

  err_valid = sqrt(sum((pred_out- rating).^2)/NN);
  fprintf(1, 'epoch %4i batch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n', ...
              epoch, batch, err_train(maxepoch*(i-1)+epoch), err_valid);
          
  sigmaI(epoch) = sqrt(sum((ratings_a-sum(w1_M1(aa_ma,:).*w1_P1(aa_pa,:),2)-mean_rating).^2/(numbatches*N)));
  
  sigmaU(epoch) = sqrt((sum(sum(w1_M1.*w1_M1)) + sum(sum(w1_P1.*w1_P1)))/...
            (num_p*num_feat+num_m*num_feat));          
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
  epoch = 1;

%   lambda = (sigmaI/sigmaU)^2;
end
(sigmaI(end)/sigmaU(end))^2
subplot 221
plot(err_train,'b-+','LineWidth',2)
subplot 222
plot(sigmaI,'b-+','LineWidth',2)
title('\sigma')
subplot 223
plot(sigmaU,'b-+','LineWidth',2)
title('\sigma_U')