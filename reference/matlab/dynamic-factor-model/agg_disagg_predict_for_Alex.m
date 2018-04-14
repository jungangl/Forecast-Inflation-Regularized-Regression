%% Formatting

format long g;

clear;
clc;


%% Load Data and Define Variables

series = csvread('agg_disagg_data.csv');
num_disagg_series = 10;

Yall = series(:,1);
Xall = series(:,2:num_disagg_series+1);

T = size(Yall,1);

%% Define Details of out of sample forecast experiment

h = 1; % Forecast horizon
J = 100; % Last value of Y used in first estimation is Y(J). First value forecast is Y(J+h)

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

disp('MSE from OLS AR Model');
disp(RMSE_ar);

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

disp('MSE from OLS Regression Including all Variables');
disp(RMSE_ols);


%% Forecast using Equal Weights Model Averaging. Each model has lagged Aggregate
% Variable and one Lagged Disaggregate Variable.

y_fore_ewma = [];

for j = J:1:T-h
    
    Y_j = Yall(h+1:j,1);
    
    y_fore_q = 0;
    num_model = 0;
    
    for q = 1:num_disagg_series
        
        X_j = [ones(j-h,1) Yall(1:j-h,1) Xall(1:j-h,q)];
        
        betahat = regress(Y_j,X_j);
        
        X_now = [1 Yall(j) Xall(j,q)];
        
        y_fore_q = y_fore_q + (X_now*betahat);
        
        num_model = num_model+1;
    
    end
    
    y_fore_ewma = [y_fore_ewma;(y_fore_q/num_model)];

end
    
f_err_ewma = (Yall(J+h:end)-y_fore_ewma);
RMSE_ewma = sqrt(mean(f_err_ewma.^2));

disp('MSE from Equal Weights Model Averaging');
disp(RMSE_ewma);


%% Forecast using a Dynamic Factor Model. Forecasting model has lagged aggregate
% variable and lagged values of first "r" principal components (factors). 

r = 1; % Number of dynamic factors to use

y_fore_dfm = [];

for j = J:1:T-h
    
    X_j = Xall(1:j-h,1:end);
    
    X_j_c = detrend(X_j,'constant');

    %Compute Weight Matrix by Hand:
    covx = cov(X_j_c);    
    [W,D] = eig(covx);
    W = fliplr(W);
    
    % Compute Weight Matrix using Matlab PCA command        
    %W = pca(X_j);
    
    % Compute all factors
    Factors = X_j_c*W;
    
    % Save first r factors
    Factors = Factors(:,1:r);
    
    Y_j = Yall(h+1:j,1);
    X_j = [ones(j-h,1) Yall(1:j-h,1) Factors];
    
    betahat = regress(Y_j,X_j);

    % Compute dynamic factors to form forecast
    
    X_j = Xall(1:j,1:end);
    
    X_j_c = detrend(X_j,'constant');
    
    % Compute Weight Matrix using Matlab PCA command        
    W = pca(X_j);
    
    % Compute all factors
    Factors = X_j_c*W;
    
    % Save first r factors
    Factors = Factors(:,1:r);
    
    X_now = [1 Yall(j) Factors(j,:)];
    
    y_fore_dfm = [y_fore_dfm;(X_now*betahat)];

end

f_err_dfm = (Yall(J+h:end)-y_fore_dfm);
RMSE_dfm = sqrt(mean(f_err_dfm.^2));

disp('MSE from DFM Regression');
disp(RMSE_dfm);