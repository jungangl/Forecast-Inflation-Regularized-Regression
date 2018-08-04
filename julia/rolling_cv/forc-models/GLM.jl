###########################################################################################
#Forecast using regulated regression methods which include ridge models and lasso models.
#This is the inner function to compute the RMSE for each λ
###########################################################################################
@everywhere function glmnet_inner_RMSE(J, h, h_lag, J_cv, Y, Y_lag, X_lag, α, λ, oos_i)
    println("oos_i = $oos_i, λ = $λ")
    ## Vector Index for oos forecast goes from J_cv - (h_lag + h) to J - (1 + h_lag + h)
    ŷ_cv = zeros(Float64, size(J_cv - (h_lag + h):J + (oos_i - 1) - (1 + h_lag + h)))
    for (n, vec_î) in enumerate(J_cv - (h_lag + h):J + (oos_i - 1) - (1 + h_lag + h))
        LHS = Y[1:vec_î - h]
        sample_size = length(LHS)
        RHS = [Y_lag[1:sample_size] X_lag[1:sample_size, :]]
        res = glmnet(RHS, LHS, alpha = α, lambda = [λ], tol = 1e-16, maxit = 10_000_000)
        β̂ = [res.a0; collect(res.betas)]
        RHS_crnt = [1; Y_lag[vec_î]; X_lag[vec_î, :]]
        ŷ_cv[n] = dot(β̂, RHS_crnt)
    end
    RMSE = sqrt(mean((Y[J_cv - (h_lag + h):J + (oos_i - 1) - (1 + h_lag + h)] - ŷ_cv) .^ 2))
    println("RMSE = $RMSE")
    return [RMSE, λ]
end



###########################################################################################
#Forecast using regulated regression methods which include ridge models and lasso models.
#This is the inner function to compute the RMSE for each λ
###########################################################################################
function glmneoos_i_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, λ_vec, α, oos_i)
    # Step1 : pick the best λ using parallel computing
    RMSE_cv = zeros(length(λ_vec), 2)
    result = pmap(λ -> glmnet_inner_RMSE(J, h, h_lag, J_cv, Y, Y_lag, X_lag, α, λ, oos_i),
                  λ_vec)::Array{Array{Float64,1},1}
    for i in 1:length(λ_vec)
        RMSE_cv[i, :] = result[i]
    end
    index = findmin(RMSE_cv[:, 1])[2]
    λ = λ_vec[index]
    # Step 2: oos forecasting
    ## Vector Index for oos forecast goes from J - (h_lag + h) to length(Y)
    vec_î = (J - (h_lag + h):length(Y))[oos_i]
    LHS = Y[1:vec_î - h]
    sample_size = length(LHS)
    RHS = [Y_lag[1:sample_size] X_lag[1:sample_size, :]]
    res = glmnet(RHS, LHS, alpha = α, lambda = [λ], tol = 10e-16, maxit = 10_000_000)
    β̂ = [res.a0; collect(res.betas)]
    RHS_crnt = [1; Y_lag[vec_î]; X_lag[vec_î, :]]
    ŷ = dot(β̂, RHS_crnt)
    return β̂, λ, ŷ, RMSE_cv
end
