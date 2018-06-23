#############################################################

## Model Average out of sample estimation,
## Start with observation from t = 1 to J
## Return the forecast values from t = J+1 to T
function modelavg_oos_forc(J, h, Q, Y, Y_lag, X_lag)
    N = size(X_lag, 2)
    T_forc = length((J + h):length(Y))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(Y))
        picks = collect(combinations((1:N), Q))
        y_forcs = zeros(length(picks))
        for n in 1:length(picks)
            LHS = Y[1:(t - h)]
            RHS = [ones(length(LHS)) Y_lag[1:length(LHS)]]
            RHS = hcat(RHS, X_lag[1:length(LHS), picks[n]])
            βhat = OLSestimator(LHS, RHS)
            x_crnt = [1.0; Y_lag[t]; X_lag[t, picks[n]]]
            y_forcs[n] = (βhat' * x_crnt)[1]
        end
        ŷ[i] = mean(y_forcs)
    end
    return ŷ
end



## Model Average with interactions out of sample estimation,
## Start with observation from t = 1 to J
## Return the forecast values from t = J+1 to T
function modelavg_oos_forc(J, h, Q, Y, Y_lag, X_lag, X_lag2)
    X_lag_comb = [X_lag X_lag2]
    N = size(X_lag_comb, 2)
    T_forc = length((J + h):length(Y))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(Y))
        picks = collect(combinations((1:N), Q))
        y_forcs = zeros(length(picks))
        for n in 1:length(picks)
            LHS = Y[1:(t - h)]
            RHS = [ones(length(LHS)) Y_lag[1:length(LHS)]]
            RHS = hcat(RHS, X_lag_comb[1:length(LHS), picks[n]])
            βhat = OLSestimator(LHS, RHS)
            x_crnt = [1.0; Y_lag[t]; X_lag_comb[t, picks[n]]]
            y_forcs[n] = (βhat' * x_crnt)[1]
        end
        ŷ[i] = mean(y_forcs)
    end
    return ŷ
end
