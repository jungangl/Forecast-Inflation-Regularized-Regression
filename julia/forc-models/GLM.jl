###########################################################################################
#Forecast using regulated regression methods which include ridge models and lasso models.
#This is the inner function to compute the RMSE for each λ
###########################################################################################
@everywhere function glmnet_inner_RMSE(J, h, h_lag, J_cv, Y, Y_lag, X_lag, α, λ)
    ## Vector Index for oos forecast goes from J_cv - (h_lag + h) to J - (1 + h_lag + h)
    ŷ_cv = zeros(Float64, size(J_cv - (h_lag + h):J - (1 + h_lag + h)))
    for (n, vec_î) in enumerate(J_cv - (h_lag + h):J - (1 + h_lag + h))
        LHS = Y[1:vec_î - h]
        sample_size = length(LHS)
        RHS = [Y_lag[1:sample_size] X_lag[1:sample_size, :]]
        res = glmnet(RHS, LHS, alpha = α, lambda = [λ], tol = 10e-20, maxit = 10_000_000)
        β̂ = [res.a0; collect(res.betas)]
        RHS_crnt = [1; Y_lag[vec_î]; X_lag[vec_î, :]]
        ŷ_cv[n] = dot(β̂, RHS_crnt)
    end
    RMSE = sqrt(mean((Y[J_cv - (h_lag + h):J - (1 + h_lag + h)] - ŷ_cv) .^ 2))
    return [RMSE, λ]
end



###########################################################################################
#Forecast using regulated regression methods which include ridge models and lasso models.
#This is the inner function to compute the RMSE for each λ
###########################################################################################
function glmnet_oos_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, λ_vec, α)
    β̂ = ones(size(X_lag, 2) + 2, 1)
    # Step1 : pick the best λ using parallel computing
    RMSE_cv = zeros(length(λ_vec), 2)
    result = pmap(λ -> glmnet_inner_RMSE(J, h, h_lag, J_cv, Y, Y_lag, X_lag, α, λ),
                  λ_vec)::Array{Array{Float64,1},1}
    for i in 1:length(λ_vec)
        RMSE_cv[i, :] = result[i]
    end
    index = findmin(RMSE_cv[:, 1])[2]
    λ = λ_vec[index]
    # Step 2: oos forecasting
    ## Vector Index for oos forecast goes from J - (h_lag + h) to length(Y)
    ŷ = zeros(Float64, size(J - (h_lag + h):length(Y)))
    for (n, vec_î) in enumerate(J - (h_lag + h):length(Y))
        LHS = Y[1:vec_î - h]
        sample_size = length(LHS)
        RHS = [Y_lag[1:sample_size] X_lag[1:sample_size, :]]
        res = glmnet(RHS, LHS, alpha = α, lambda = [λ], tol = 10e-20, maxit = 10_000_000)
        β̂ = [res.a0; collect(res.betas)]
        RHS_crnt = [1; Y_lag[vec_î]; X_lag[vec_î, :]]
        ŷ[n] = dot(β̂, RHS_crnt)
    end
    RMSE = sqrt(mean((Y[J - (h_lag + h):length(Y)] - ŷ) .^ 2))
    return β̂, λ, ŷ, RMSE, RMSE_cv
end

#α = 0.0
#λ_vec = 10.:0.1:20. or λ_vec = 0.1.:0.2:20.
#β̂, λ, ŷ, RMSE_cv = glmnet_oos_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, λ_vec, α)
#plot(RMSE_cv[:, 1])
#plot(Y[J - (h_lag + h):length(Y)])
#plot!(ŷ)
#RMSE = sqrt(mean((Y[J - (h_lag + h):length(Y)] - ŷ) .^ 2))
#println("Tuning Parameter Chosen for ridge: $λ")
#println("RMSE from ridge including first order terms: \nRMSE = $RMSE")
