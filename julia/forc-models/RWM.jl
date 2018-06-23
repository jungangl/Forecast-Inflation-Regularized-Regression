#############################################################
#=
Forecast ssing Random Walk Model
It is a special case of AR1 model with AR parameter set to be 1
Start with observation from t = 1 to J
Return the forecast values from t = J+1 to T
This model can be written as
y(t) = α + y(t-6) + ε(t-6)
We can regress
y(t) - y(t-6) on a vector of 1's to get the estimate for the drift α̂
=#
function rwm_oos_forc(J, h, Y, Y_lag)
    T_forc = length((J + h):length(Y))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(Y))
        LHS = Y[1:(t - h)]
        LHS_lag = Y_lag[1:length(LHS)]
        α̂ = OLSestimator(LHS - LHS_lag, ones(length(LHS)))
        y_crnt = [1.0; Y_lag[t]]
        ŷ[i] = dot(y_crnt, [α̂ ; 1.0])
    end
    return ŷ
end
