function run_BMA(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2)
    df = DataFrame()
    name = "BMA"
    folder_name = "result-forc-indi"
    df_file = "../../../data/$folder_name/level$level-h$h-J$J/$name.csv"
    ## Forecast using Bayesian Model Averaging
    θ_vec = 0.05:0.01:0.99
    pre_conv = 1_000
    post_conv = 2_000
    θ, ŷ, RMSE, RMSE_cv = bma_oos_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, θ_vec, pre_conv, post_conv)
    println("Prior parameter θ chosen for $(name): $θ")
    println("RMSE from Bayesian Model Averaging, h = $h: \nRMSE = $RMSE")
    println("RMSE_cv is $RMSE_cv")
    df[Symbol(name)] = ŷ
    CSV.write(df_file, df)
end



function run_BMA2(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2)
    df = DataFrame()
    name = "BMA2"
    folder_name = "result-forc-indi"
    df_file = "../../../data/$folder_name/level$level-h$h-J$J/$name.csv"
    ## Forecast using Bayesian Model Averaging with interaction terms
    θ_vec = 0.05:0.01:0.99
    pre_conv = 1_000
    post_conv = 2_000
    X_lag = hcat(X_lag, X_lag2)
    θ, ŷ, RMSE, RMSE_cv = bma_oos_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, θ_vec, pre_conv, post_conv)
    println("Prior parameter θ chosen for $(name) with interactions: $θ")
    println("RMSE from Bayesian Model Averaging with interaction terms, h = $h: \nRMSE = $RMSE")
    println("RMSE_cv is $RMSE_cv")
    df[Symbol(name)] = ŷ
    CSV.write(df_file, df)
end



function run_DFM2(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2)
    df = DataFrame()
    name = "DFM2"
    folder_name = "result-forc-indi"
    df_file = "../../../data/$folder_name/level$level-h$h-J$J/$name.csv"
    #Forecast using a Dynamic Factor Model with interaction terms
    r1 = 1
    r2 = 1
    ŷ, RMSE = dfm_oos_forc(J, h, h_lag, Y, Y_lag, X_lag, X_lag2, r1, r2)
    println("Dynamic Factor with interaction terms r1 = $(r1), r2 = $(r2): \nRMSE = $RMSE")
    df[Symbol(name)] = ŷ
    CSV.write(df_file, df)
end



function run_LAS(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2)
    df = DataFrame()
    name = "LAS"
    folder_name = "result-forc-indi"
    df_file = "../../../data/$folder_name/level$level-h$h-J$J/$name.csv"
    #Forecast using a LASSO model
    α = 1.0
    λ_vec = 0.1:0.1:5.
    β̂, λ, ŷ, RMSE, RMSE_cv = glmnet_oos_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, λ_vec, α)
    println("Tuning Parameter Chosen for lasso: $λ")
    println("Percent of Slope Coefficients Set Equal to Zero by lasso: $(round(100 * (sum(β̂[2:end] .==  .0) / length(β̂[2:end])), 2))")
    println("RMSE from lasso Including all Variables, h = $h: \nRMSE = $RMSE")
    println("RMSE_cv is $RMSE_cv")
    df[Symbol(name)] = ŷ
    CSV.write(df_file, df)
end



function run_LAS2(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2)
    df = DataFrame()
    name = "LAS2"
    folder_name = "result-forc-indi"
    df_file = "../../../data/$folder_name/level$level-h$h-J$J/$name.csv"
    #Forecast using a LASSO model with interaction terms
    α = 1.0
    λ_vec = 0.1:0.1:5.
    X_lag = hcat(X_lag, X_lag2)
    β̂, λ, ŷ, RMSE, RMSE_cv = glmnet_oos_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, λ_vec, α)
    println("Tuning Parameter Chosen for lasso: $λ")
    println("Percent of Slope Coefficients Set Equal to Zero by lasso: $(round(100 * (sum(β̂[2:end] .==  .0) / length(β̂[2:end])), 2))")
    println("RMSE from lasso Including all Variables, h = $h: \nRMSE = $RMSE")
    println("RMSE_cv is $RMSE_cv")
    df[Symbol(name)] = ŷ
    CSV.write(df_file, df)
end



function run_RDG(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2)
    df = DataFrame()
    name = "RDG"
    folder_name = "result-forc-indi"
    df_file = "../../../data/$folder_name/level$level-h$h-J$J/$name.csv"
    #Forecast using a Ridge model with interaction terms
    α = 0.0
    λ_vec = 10:0.1:20
    β̂, λ, ŷ, RMSE, RMSE_cv = glmnet_oos_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, λ_vec, α)
    println("Tuning Parameter Chosen for ridge: $λ")
    println("RMSE from ridge including first order terms: \nRMSE = $RMSE")
    println("RMSE_cv is $RMSE_cv")
    df[Symbol(name)] = ŷ
    CSV.write(df_file, df)
end



function run_RDG2(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2)
    df = DataFrame()
    name = "RDG2"
    folder_name = "result-forc-indi"
    df_file = "../../../data/$folder_name/level$level-h$h-J$J/$name.csv"
    #Forecast using a Ridge model with interaction terms
    α = 0.0
    λ_vec = 10:0.1:20
    X_lag = hcat(X_lag, X_lag2)
    β̂, λ, ŷ, RMSE, RMSE_cv = glmnet_oos_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, λ_vec, α)
    println("Tuning Parameter Chosen for ridge: $λ")
    println("RMSE from ridge including second order terms: \nRMSE = $RMSE")
    println("RMSE_cv is $RMSE_cv")
    df[Symbol(name)] = ŷ
    CSV.write(df_file, df)
end



function run_RDF(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2)
    df = DataFrame()
    name = "RDF"
    folder_name = "result-forc-indi"
    df_file = "../../../data/$folder_name/level$level-h$h-J$J/$name.csv"
    #Forecast using a Random Forrest Regression model
    m_vec = 1:1:25
    n_est = 50
    m, ŷ, RMSE, RMSE_cv = rfr_oos_forc(J , h, h_lag, J_cv, Y, Y_lag, X_lag, m_vec, n_est)
    println("Depth parameter chosen for Random Forest: $m")
    println("RMSE from Random Forest Including all Variables, h = $h: \nRMSE = $RMSE")
    df[Symbol(name)] = ŷ
    CSV.write(df_file, df)
end



function run_RDF2(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2)
    df = DataFrame()
    name = "RDF2"
    folder_name = "result-forc-indi"
    df_file = "../../../data/$folder_name/level$level-h$h-J$J/$name.csv"
    #Forecast using a Random Forrest Regression model
    m_vec = 1:1:25
    n_est = 50
    X_lag = hcat(X_lag, X_lag2)
    m, ŷ, RMSE, RMSE_cv = rfr_oos_forc(J , h, h_lag, J_cv, Y, Y_lag, X_lag, m_vec, n_est)
    println("Depth parameter chosen for Random Forest with interaction terms: $m")
    println("RMSE from Random Forest Including all Variables, h = $h: \nRMSE = $RMSE")
    df[Symbol(name)] = ŷ
    CSV.write(df_file, df)
end



function run_TRD(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2)
    df = DataFrame()
    name = "TRD"
    folder_name = "result-forc-indi"
    df_file = "../../../data/$folder_name/level$level-h$h-J$J/$name.csv"
    df[:REAL] = Y[J - (h_lag + h):end]

    #Forecast using the dynamic factor model
    r = 1
    ŷ, RMSE = dfm_oos_forc(J, h, h_lag, Y, Y_lag, X_lag, r)
    println("Dynamic Factor r = $r: \nRMSE = $RMSE")
    df[:DFM] = ŷ

    #Forecast using AR1 model #
    ŷ, RMSE = AR_oos_forc(J, h, h_lag, Y, Y_lag)
    println("AR(1) Model: \nRMSE = $RMSE")
    df[:ARM] =  ŷ

    #Forecast using OLS with Lagged Aggregate Variable
    ŷ, RMSE = OLS_oos_forc(J, h, h_lag, Y, Y_lag, X_lag)
    println("OLS Regression Including all Variables: \nRMSE = $RMSE")
    df[:OLS] = ŷ

    #Model Average out of sample estimation
    Q = 1
    ŷ, RMSE = modelavg_oos_forc(J, h, h_lag, Y, Y_lag, X_lag, Q)
    println("Equal Weights Model Averaging with Q = $Q: \nRMSE = $RMSE")
    df[:MAG] = ŷ

    #Model Average with interactions
    Q = 1
    X_lag = hcat(X_lag, X_lag2)
    ŷ, RMSE = modelavg_oos_forc(J, h, h_lag, Y, Y_lag, X_lag, Q)
    println("Equal Weights Model Averaging with interactions, Q = $Q: \nRMSE = $RMSE")
    df[:MAG2] = ŷ

    #Forecast using the random walk model
    ŷ, RMSE = rwm_oos_forc(J, h, h_lag, Y, Y_lag)
    println("Random Walk Model: \nRMSE = $RMSE")
    df[:RWM] = ŷ
    CSV.write(df_file, df)
end
