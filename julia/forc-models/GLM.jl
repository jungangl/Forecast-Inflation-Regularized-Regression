#############################################################
#=
Forecast using Ridge Regression
with Lagged Aggregate Variable and
Lagged Disaggregate Variables
=#
@everywhere function glmnet_inner_RMSE(J, h, J_cv, Y, Y_lag, X_lag, α, λ)
    ŷ_cv = zeros(length(J_cv + h:J))
    for (j, t) in enumerate(J_cv + h:J)
        LHS = Y[1:t - h]
        RHS = [Y_lag[1:length(LHS)] X_lag[1:length(LHS), :]]
        res = glmnet(RHS, LHS, alpha = α, lambda = [λ], tol = 10e-20)
        βhat = [res.a0; collect(res.betas)]
        x_crnt = [1; Y_lag[t]; X_lag[t, :]]
        ŷ_cv[j] = (βhat' * x_crnt)[1]
    end
    return [sqrt(mean((Y[J_cv + h:J] - ŷ_cv) .^ 2)), λ]
end


function glmnet_oos_forc(J, h, J_cv, Y, Y_lag, X_lag, λ_vec, α)
    βhat = []
    # Step1 : pick the best λ using parallel computing
    RMSE_glmnet_CV = zeros(length(λ_vec), 2)
    result = pmap(λ -> glmnet_inner_RMSE(J, h, J_cv, Y, Y_lag, X_lag, α, λ), λ_vec)
    for i in 1:length(λ_vec)
        RMSE_glmnet_CV[i, :] = result[i]
    end
    index = findmin(RMSE_glmnet_CV[:, 1])[2]
    λ = λ_vec[index]
    # Step 2: oos forecasting
    T_forc = length(J + h:length(Y))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(Y))
        LHS = Y[1:t - h]
        RHS = [Y_lag[1:length(LHS)] X_lag[1:length(LHS), :]]
        res = glmnet(RHS, LHS, alpha = α, lambda = [λ], maxit = 10_000_000)
        βhat = [res.a0; collect(res.betas)]
        x_crnt = [1; Y_lag[t]; X_lag[t, :]]
        ŷ[i] = (βhat' * x_crnt)[1]
    end
    return βhat, λ, ŷ
end
