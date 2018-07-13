###########################################################################################
#Forecast using random forest regression with lagged aggregate variable
#and lagged disaggregate variables.
#This is the inner function to compute the RMSE for each λ
###########################################################################################
@everywhere function rfr_inner_RMSE(J, h, h_lag, J_cv, Y, Y_lag, X_lag, n_est, m)
    ## Vector Index for oos forecast goes from J_cv - (h_lag + h) to J - (1 + h_lag + h)
    ŷ_cv = zeros(Float64, size(J_cv - (h_lag + h):J - (1 + h_lag + h)))
    model = RandomForestRegressor(n_est, max_depth = m, random_state = 0)::PyCall.PyObject
    for (n, vec_î) in enumerate(J_cv - (h_lag + h):J - (1 + h_lag + h))
        LHS = Y[1:vec_î - h]
        sample_size = length(LHS)
        RHS = [Y_lag[1:sample_size] X_lag[1:sample_size, :]]
        fit!(model, RHS, LHS)
        RHS_crnt = [Y_lag[vec_î]; X_lag[vec_î, :]]'
        ŷ_cv[n] = ScikitLearn.predict(model, RHS_crnt)[1]::Float64
    end
    return [sqrt(mean((Y[J_cv - (h_lag + h):J - (1 + h_lag + h)] - ŷ_cv) .^ 2)), m]
end



###########################################################################################
#Forecast using random forest regression with lagged aggregate variable
#and lagged disaggregate variables.
#This is the outer function to compute the RMSE for each λ
###########################################################################################
function rfr_oos_forc(J , h, h_lag, J_cv, Y, Y_lag, X_lag, m_vec, n_est)
    # Step1 : pick the best m
    RMSE_cv = zeros(length(m_vec), 2)
    result = pmap(m -> rfr_inner_RMSE(J, h, h_lag, J_cv, Y, Y_lag, X_lag, n_est, m),
                  m_vec)::Array{Array{Float64,1},1}
    for i in 1:length(m_vec)
        RMSE_cv[i,:] = result[i]
    end
    index = findmin(RMSE_cv[:, 1])[2]::Int64
    m = m_vec[index]::Int64
    model = RandomForestRegressor(n_est, max_depth = m, random_state = 0)::PyCall.PyObject
    # Step 2: oos forecasting
    ## Vector Index for oos forecast goes from J - (h_lag + h) to length(Y)
    ŷ = zeros(Float64, size(J - (h_lag + h):length(Y)))
    for (n, vec_î) in enumerate(J - (h_lag + h):length(Y))
        LHS = Y[1:vec_î - h]
        sample_size = length(LHS)
        RHS = [Y_lag[1:sample_size] X_lag[1:sample_size, :]]
        fit!(model, RHS, LHS)
        RHS_crnt = [Y_lag[vec_î]; X_lag[vec_î, :]]'
        ŷ[n] = ScikitLearn.predict(model, RHS_crnt)[1]
    end
    return m, ŷ, RMSE_cv
end
