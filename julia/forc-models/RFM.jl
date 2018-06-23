#############################################################
## Forecast using Random Forest Regression
## with Lagged Aggregate Variable and
## Lagged Disaggregate Variables

@everywhere function rfr_inner_RMSE(J, h, J_cv, Y, Y_lag, X_lag, n_est, m)
    model = RandomForestRegressor(n_est, max_depth = m, random_state = 0)
    ŷ_cv = zeros(length(J_cv + h:J))
    for (j, t) in enumerate(J_cv + h:J)
        LHS = Y[1:t - h]
        RHS = [Y_lag[1:length(LHS)] X_lag[1:length(LHS), :]]
        fit!(model, RHS, LHS)
        x_crnt = [Y_lag[t]; X_lag[t, :]]'
        ŷ_cv[j] = ScikitLearn.predict(model, x_crnt)[1]
    end
    return [sqrt(mean((Y[J_cv + h:J] - ŷ_cv) .^ 2)), m]
end


function rfr_oos_forc(J , h, J_cv, Y, Y_lag, X_lag, m_vec; n_est = 50)
    # Step1 : pick the best m
    RMSE_rfr_CV = zeros(length(m_vec), 2)
    result = pmap(m -> rfr_inner_RMSE(J, h, J_cv, Y, Y_lag, X_lag, n_est, m), m_vec)
    for i in 1:length(m_vec)
        RMSE_rfr_CV[i,:] = result[i]
    end
    index = findmin(RMSE_rfr_CV[:, 1])[2]
    m = m_vec[index]
    model = RandomForestRegressor(n_est, max_depth = m, random_state = 0)
    # Step 2: oos forecasting
    T_forc = length(J + h:length(Y))
    ŷ = zeros(T_forc)
    println("Start computing oos RMSE")
    for (i, t) in enumerate((J + h):length(Y))
        LHS = Y[1:t - h]
        RHS = [Y_lag[1:length(LHS)] X_lag[1:length(LHS), :]]
        fit!(model, RHS, LHS)
        x_crnt = [Y_lag[t]; X_lag[t, :]]'
        ŷ[i] = ScikitLearn.predict(model, x_crnt)[1]
    end
    return m, ŷ
end


## Forecast using Random Forest Regression with interaction terms
## with Lagged Aggregate Variable and
## Lagged Disaggregate Variables


function rfr_oos_forc(J , h, J_cv, Y, Y_lag, X_lag, X_lag2, m_vec; n_est = 50)
    X_lag = [X_lag X_lag2]
    # Step1 : pick the best m
    RMSE_rfr_CV = zeros(length(m_vec), 2)
    result = pmap(m -> rfr_inner_RMSE(J, h, J_cv, Y, Y_lag, X_lag, n_est, m), m_vec)
    for i in 1:length(m_vec)
        RMSE_rfr_CV[i,:] = result[i]
    end
    index = findmin(RMSE_rfr_CV[:, 1])[2]
    m = m_vec[index]
    model = RandomForestRegressor(n_est, max_depth = m, random_state = 0)
    # Step 2: oos forecasting
    T_forc = length(J + h:length(Y))
    ŷ = zeros(T_forc)
    println("Start computing oos RMSE")
    for (i, t) in enumerate((J + h):length(Y))
        LHS = Y[1:t - h]
        RHS = [Y_lag[1:length(LHS)] X_lag[1:length(LHS), :]]
        fit!(model, RHS, LHS)
        x_crnt = [Y_lag[t]; X_lag[t, :]]'
        ŷ[i] = ScikitLearn.predict(model, x_crnt)[1]
    end
    return m, ŷ
end
