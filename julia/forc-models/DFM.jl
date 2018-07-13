#########################################################################
##Define the function that extracts the first r factors of
##the data in the X matrix from time 1 to time t
#########################################################################
function get_factors(vec_î, X_lag, r)
    x_j = X_lag[1:vec_î, :]
    x_j_c = x_j .- mean(x_j, 1)
    # computing the weight matrix
    covx = cov(x_j_c)
    # W is the weight matrix
    F = eig(covx)[2]
    W = flipdim(F, 2)::Array{Float64, 2}
    res = (x_j_c * W)[:, 1:r]
    return res
end


#########################################################################
#Forecast using a dynamic factor model with lagged aggregate variables
#and lagged values of first "r" principal components (factors).
#########################################################################
function dfm_oos_forc(J, h, h_lag, Y, Y_lag, X_lag, r)
    ## Vector Index for oos forecast goes from J - (h_lag + h) to length(Y)
    ŷ = zeros(Float64, size(J - (h_lag + h):length(Y)))
    for (n, vec_î) in enumerate(J - (h_lag + h):length(Y))
        LHS = Y[1:(vec_î - h)]
        sample_size = length(LHS)
        factors = get_factors(vec_î, X_lag, r)
        RHS = [ones(sample_size) Y_lag[1:sample_size] factors[1:sample_size, :]]
        β̂ = OLSestimator(LHS, RHS)
        # Compute Forecasts
        RHS_crnt = [1; Y_lag[vec_î]; factors[vec_î, :]]
        ŷ[n] = dot(β̂, RHS_crnt)
    end
    RMSE = sqrt(mean((Y[J - (h_lag + h):length(Y)] - ŷ) .^ 2))
    return ŷ, RMSE
end



#########################################################################
#Forecast using a dynamic factor model with lagged aggregate variables
#and lagged values of first 1 principal components from
#1) the first order terms and
#2) the second order interactions terms.
#########################################################################
function dfm_oos_forc(J, h, h_lag, Y, Y_lag, X_lag, X_lag2, r1, r2)
    ## Vector Index for oos forecast goes from J - (h_lag + h) to length(Y)
    ŷ = zeros(Float64, size(J - (h_lag + h):length(Y)))
    for (n, vec_î) in enumerate(J - (h_lag + h):length(Y))
        println(n)
        LHS = Y[1:(vec_î - h)]
        sample_size = length(LHS)
        factors1 = get_factors(vec_î, X_lag, r1)
        factors2 = get_factors(vec_î, X_lag2, r2)
        RHS = [ ones(sample_size) Y_lag[1:sample_size]
                factors1[1:sample_size, :] factors2[1:sample_size, :]
                ]
        β̂ = OLSestimator(LHS, RHS)
        # Compute Forecasts
        RHS_crnt = [1; Y_lag[vec_î]; factors1[vec_î, :]; factors2[vec_î, :]]
        ŷ[n] = dot(β̂, RHS_crnt)
    end
    RMSE = sqrt(mean((Y[J - (h_lag + h):length(Y)] - ŷ) .^ 2))
    return ŷ, RMSE
end
