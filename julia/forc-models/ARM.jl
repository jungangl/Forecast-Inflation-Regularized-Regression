#####################################################################################
#Forecast using OLS with only lagged aggregate variable
#The first out-of-sample forecast has vector index of J - (h_lag + h)
#####################################################################################
function AR_oos_forc(J, h, h_lag, Y, Y_lag)
    ŷ = zeros(Float64, size(J - (h_lag + h):length(Y)))
    for (n, vec_î) in enumerate(J - (h_lag + h):length(Y))
        LHS = Y[1:(vec_î - h)]
        sample_size = length(LHS)
        RHS = [ones(sample_size) Y_lag[1:sample_size]]
        β̂ = OLSestimator(LHS, RHS)
        RHS_crnt = [1.0; Y_lag[vec_î]]
        ŷ[n] = dot(β̂, RHS_crnt)
    end
    RMSE = sqrt(mean((Y[J - (h_lag + h):length(Y)] - ŷ) .^ 2))
    return ŷ, RMSE
end
