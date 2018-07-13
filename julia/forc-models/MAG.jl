#####################################################################################
#Forecast using model averaging method with lagged aggregate variable.
#Averging model of size Q
#####################################################################################
function modelavg_oos_forc(J, h, h_lag, Y, Y_lag, X_lag, Q)
    ## Vector Index for oos forecast goes from J - (h_lag + h) to length(Y)
    ŷ = zeros(Float64, size(J - (h_lag + h):length(Y)))
    N = size(X_lag, 2)
    picks = collect(combinations((1:N), Q))
    ŷ_picks = zeros(length(picks))
    for (n, vec_î) in enumerate(J - (h_lag + h):length(Y))
        for (p, pick) in enumerate(picks)
            LHS = Y[1:(vec_î - h)]
            sample_size = length(LHS)
            RHS = hcat([ones(sample_size) Y_lag[1:sample_size]], X_lag[1:sample_size, pick])
            β̂ = OLSestimator(LHS, RHS)
            RHS_crnt = [1.0; Y_lag[vec_î]; X_lag[vec_î, pick]]
            ŷ_picks[p] = dot(β̂, RHS_crnt)
        end
        ŷ[n] = mean(ŷ_picks)
    end
    RMSE = sqrt(mean((Y[J - (h_lag + h):length(Y)] - ŷ) .^ 2))
    return ŷ, RMSE
end
