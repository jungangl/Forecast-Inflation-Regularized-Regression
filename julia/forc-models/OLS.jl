#############################################################
#=
Forecast Using OLS with Lagged Aggregate Variable
And Lagged Disaggregate Variable
Leaving out the last disaggregate variable
Start with observation from t = 1 to J
Return the forecast values from t = J+1 to T
=#
function OLS_oos_forc(J, h, Y, Y_lag, X_lag)
    T_forc = length((J + h):length(Y))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(Y))
        LHS = Y[1:(t - h)]
        RHS = [ones(length(LHS)) Y_lag[1:length(LHS)] X_lag[1:length(LHS), :]]
        βhat = OLSestimator(LHS, RHS)
        x_crnt = [1.0; Y_lag[t]; X_lag[t, :]]
        ŷ[i] = (βhat' * x_crnt)[1]
    end
    return ŷ
end
