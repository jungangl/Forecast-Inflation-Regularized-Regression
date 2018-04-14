%% Formatting

format long;

clear;
clc;


%% Load Data and Define Variables

series = csvread('agg_disagg_data.csv');
num_disagg_series = 10;

Yall = series(:,1);
Xall = series(:,2:num_disagg_series+1);

T = size(Yall,1);


%% Define Details of out of sample forecast experiment

h = 2; % Forecast horizon
J = 100; % Out-of-sample forecasts will be produced for Y(J+h) through Y(T)
    
CV_J = 50; % Time series cross-validation is used to set tuning parameters for 
           % machine learning techniques. Last value of Y used in first 
           % training sample is Y(CV_J). First evaluation period is Y(CV_J+h). 
           % Last evaluation period is Y(J) 


%% Forecast Using OLS with Only Lagged Aggregate Variable (AR model)

y_fore_ar = [];

for j = J:1:T-h
    
    Y_j = Yall(h+1:j,1);
    X_j = [ones(j-h,1) Yall(1:j-h,1)];
    
    betahat = regress(Y_j,X_j);
    
    X_now = [1 Yall(j)];
   
    y_fore_ar = [y_fore_ar;(X_now*betahat)];
    
end

f_err_ar = (Yall(J+h:end)-y_fore_ar);
RMSE_ar = sqrt(mean(f_err_ar.^2));


%% Forecast Using OLS with Lagged Aggregate Variable and Lagged Disaggregate Variable

y_fore_ols = [];

for j = J:1:T-h
    
    Y_j = Yall(h+1:j,1);
    X_j = [ones(j-h,1) Yall(1:j-h,1) Xall(1:j-h,1:(end-1))];
    
    betahat = regress(Y_j,X_j);
    
    X_now = [1 Yall(j) Xall(j,1:(end-1))];
   
    y_fore_ols = [y_fore_ols;(X_now*betahat)];
    
end

f_err_ols = (Yall(J+h:end)-y_fore_ols);
RMSE_ols = sqrt(mean(f_err_ols.^2));


%% Forecast using Equal Weights Model Averaging applied to a limited model set.
% Each model has lagged Aggregate Variable and Q Lagged Disaggregate Variables.
% This could become computationally burdensome for some values of Q if there are 
% a large number of Disaggregate variables, as all models with Q variables are considered.

y_fore_ewma = [];

Q=1;

inc_mat = nchoosek((1:1:num_disagg_series),Q);

for j = J:1:T-h
    
    Y_j = Yall(h+1:j,1);
    
    y_fore_q = 0;
    
    for q = 1:size(inc_mat,1)
        
        X_j = [ones(j-h,1) Yall(1:j-h,1) Xall(1:j-h,inc_mat(q,:))];
        
        betahat = regress(Y_j,X_j);
        
        X_now = [1 Yall(j) Xall(j,inc_mat(q,:))];
        
        y_fore_q = y_fore_q + (X_now*betahat);
    
    end
    
    y_fore_ewma = [y_fore_ewma;(y_fore_q/size(inc_mat,1))];

end
    
f_err_ewma = (Yall(J+h:end)-y_fore_ewma);
RMSE_ewma = sqrt(mean(f_err_ewma.^2));


%% Forecast using a Dynamic Factor Model. Forecasting model has lagged aggregate
% variable and lagged values of first "r" principal components (factors). 

r = 1; % Number of dynamic factors to use

y_fore_dfm = [];

for j = J:1:T-h
        
    X_j = Xall(1:j,1:end);
    
    X_j_c = detrend(X_j,'constant');

    % Compute Weight Matrix by Hand:
    % covx = cov(X_j_c);    
    % [W,D] = eig(covx);
    % W = fliplr(W);
            
    % Compute Weight Matrix using Matlab PCA command        
    W = pca(X_j);
        
    % Compute all factors
    Factors = X_j_c*W;
    
    % Save first r factors
    Factors = Factors(:,1:r);
    
    Y_j = Yall(h+1:j,1);
    X_j = [ones(j-h,1) Yall(1:j-h,1) Factors(1:j-h,:)];
        
    betahat = regress(Y_j,X_j);
    
    % Compute o.o.s. prediction
    
    X_now = [1 Yall(j) Factors(j,:)];
    
    y_fore_dfm = [y_fore_dfm;(X_now*betahat)];

end

f_err_dfm = (Yall(J+h:end)-y_fore_dfm);
RMSE_dfm = sqrt(mean(f_err_dfm.^2));


%% Forecast using Ridge Regression with Lagged Aggregate Variable and 
%  Lagged Disaggregate Variables

y_fore_ridge = [];

% Set range of values of ridge parameter
%k = (0.01:0.01:20)';
k = (20:0.01:30)';
 
RMSE_ridge_CV = [];

disp('Estimating tuning parameter for ridge regression.')
disp('Percent Complete...')

first_print=1;

for k_indx = 1:size(k,1)
    
    perc_comp = (k_indx/size(k,1))*100;
            
    if rem(perc_comp,10)<0.1
        
        if first_print==0
            fprintf('\b\b');
        end
        
        if first_print==1
            fprintf('\b');
        end
        
        first_print=0;
        
        fprintf('%d', floor(perc_comp));
        
    end
   
    k_CV = k(k_indx);
    
    y_fore_ridge_CV = [];
    
    for j_indx = CV_J:1:J-h
        
        Y_j = Yall(h+1:j_indx,1);
        X_j = [Yall(1:j_indx-h,1) Xall(1:j_indx-h,:)];
        
        betahat_ridge = ridge(Y_j, X_j, k_CV, 0);
        
        X_now = [1 Yall(j_indx) Xall(j_indx,:)];
        
        y_fore_ridge_CV = [y_fore_ridge_CV;(X_now*betahat_ridge)];
    
    end
    
    f_err_ridge_CV = (Yall(CV_J+h:J)-y_fore_ridge_CV);
    RMSE_ridge_CV = [RMSE_ridge_CV;[sqrt(mean(f_err_ridge_CV.^2)) k(k_indx)]];
  
end

clc;

[~, indx] = min(RMSE_ridge_CV(:,1));
k_ridge = k(indx);

for j = J:1:T-h
    
    Y_j = Yall(h+1:j,1);
    X_j = [Yall(1:j-h,1) Xall(1:j-h,:)];
    
    betahat_ridge = ridge(Y_j, X_j, k_ridge, 0);
    
    X_now = [1 Yall(j) Xall(j,:)];
    
    y_fore_ridge = [y_fore_ridge;(X_now*betahat_ridge)];

end

f_err_ridge = (Yall(J+h:end)-y_fore_ridge);
RMSE_ridge = sqrt(mean(f_err_ridge.^2));


%% Forecast using lasso with Lagged Aggregate Variable and Lagged 
% Disaggregate Variables

y_fore_lasso = [];

% Set range of values of lasso tuning parameter
k = (0.01:0.01:10)';
 
RMSE_lasso_CV = [];

disp('Estimating tuning parameter for lasso.')
disp('Percent Complete...')

first_print=1;

for k_indx = 1:size(k,1)
    
    perc_comp = (k_indx/size(k,1))*100;
            
    if rem(perc_comp,10)<0.1
        
        if first_print==0
            fprintf('\b\b');
        end
        
        if first_print==1
            fprintf('\b');
        end
        
        first_print=0;
        
        fprintf('%d', floor(perc_comp));
        
    end
        
    k_CV = k(k_indx);
    
    y_fore_lasso_CV = [];
    
    for j_indx = CV_J:1:J-h
        
        Y_j = Yall(h+1:j_indx,1);
        X_j = [Yall(1:j_indx-h,1) Xall(1:j_indx-h,:)];
        
        [betahat_lasso, stats] = lasso(X_j, Y_j, 'Lambda', k_CV);
        betahat_lasso = [stats.Intercept;betahat_lasso];
        
        X_now = [1 Yall(j_indx) Xall(j_indx,:)];
        
        y_fore_lasso_CV = [y_fore_lasso_CV;(X_now*betahat_lasso)];
    
    end
    
    f_err_lasso_CV = (Yall(CV_J+h:J)-y_fore_lasso_CV);
    RMSE_lasso_CV = [RMSE_lasso_CV;[sqrt(mean(f_err_lasso_CV.^2)) k(k_indx)]];
  
end

[val,indx] = min(RMSE_lasso_CV(:,1));
k_lasso = k(indx);

for j = J:1:T-h
    
    Y_j = Yall(h+1:j,1);
    X_j = [Yall(1:j-h,1) Xall(1:j-h,:)];
    
    [betahat_lasso, stats] = lasso(X_j, Y_j, 'Lambda', k_lasso);
    betahat_lasso = [stats.Intercept;betahat_lasso];
        
    X_now = [1 Yall(j) Xall(j,:)];
    
    y_fore_lasso = [y_fore_lasso;(X_now*betahat_lasso)];

end

f_err_lasso = (Yall(J+h:end)-y_fore_lasso);
RMSE_lasso = sqrt(mean(f_err_lasso.^2));


%% Display Output for Each Model

clc;

disp('RMSE from OLS AR Model');
disp(RMSE_ar);

disp('RMSE from OLS Regression Including all Variables');
disp(RMSE_ols);

say = sprintf('RMSE from Equal Weights Model Averaging with Q = %d', Q);
disp(say);
disp(RMSE_ewma);

disp('RMSE from DFM Regression');
disp(RMSE_dfm);

say = sprintf('Tuning Parameter chosen for Ridge Regression: %f', k_ridge);
disp(say);
disp('RMSE from Ridge Regression Including all Variables');
disp(RMSE_ridge);

say = sprintf('Tuning Parameter Chosen for lasso: %f', k_lasso);
disp(say);
say = sprintf('Percent of Slope Coefficients Set Equal to Zero by lasso: %f', 100-(nnz(betahat_lasso(2:end))/size(betahat_lasso(2:end),1))*100);
disp(say);
disp('RMSE from lasso Including all Variables');
disp(RMSE_lasso);