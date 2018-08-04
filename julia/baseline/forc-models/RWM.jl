#########################################################################
#Forecast using random walk model, which is a special case of AR1 model
#with AR parameter set to be 1
#########################################################################
function rwm_oos_forc(J, h, h_lag, Y, Y_lag)
    ŷ = zeros(Float64, size(J - (h_lag + h):length(Y)))
    for (n, vec_î) in enumerate(J - (h_lag + h):length(Y))
        LHS = Y[1:(vec_î - h)]
        sample_size = length(LHS)
        LHS_lag = Y_lag[1:sample_size]
        ε̂ = mean(LHS - LHS_lag)
        ŷ[n] = Y_lag[vec_î] + ε̂
    end
    RMSE = sqrt(mean((Y[J - (h_lag + h):length(Y)] - ŷ) .^ 2))
    return ŷ, RMSE
end
