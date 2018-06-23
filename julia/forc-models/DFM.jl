##############################################################


##Define the function that extracts the first r factors of
##the data in the X matrix from time 1 to time t
function get_factors(t, X, r)
    x_j = X[1:t, :]
    x_j_c = x_j .- mean(x_j, 1)
    # computing the weight matrix
    covx = cov(x_j_c)
    # W is the weight matrix
    W = eig(covx)[2]
    W = flipdim(W,2)
    return (x_j_c * W)[:, 1:r]
end




## Forecast using a Dynamic Factor Model.
## Forecasting model has lagged aggregate
## variable and lagged values of first "r"
## principal components (factors).
function dfm_oos_forc(J, h, Y, Y_lag, X_lag; r = 1)
    T_forc = length((J + h):length(Y))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(Y))
        factors = get_factors(t, X_lag, r)
        LHS = Y[1:(t - h)]
        RHS = [ones(length(LHS)) Y_lag[1:length(LHS)] factors[1:length(LHS), :]]
        βhat = OLSestimator(LHS, RHS)
        # Compute Forecasts
        x_crnt = [1; Y_lag[t]; factors[t, :]]
        ŷ[i] = (βhat' * x_crnt)[1]
    end
    return ŷ
end



## Forecast using a Dynamic Factor Model.
## Forecasting model has lagged aggregate variable
## and lagged values of first 1 principal components
## for the first order terms and second order interactions terms each.
function dfm_oos_forc(J, h, Y, Y_lag, X_lag, X_lag2; r1 = 1, r2 = 1)
    T_forc = length((J + h):length(Y))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(Y))
        factors1 = get_factors(t, X_lag, r1)
        factors2 = get_factors(t, X_lag2, r2)
        LHS = Y[1:(t - h)]
        RHS = [ones(length(LHS)) Y_lag[1:length(LHS)] factors1[1:length(LHS), :] factors2[1:length(LHS), :]]
        βhat = OLSestimator(LHS, RHS)
        # Compute Forecasts
        x_crnt = [1; Y_lag[t]; factors1[t, :]; factors2[t, :]]
        ŷ[i] = (βhat' * x_crnt)[1]
    end
    return ŷ
end
