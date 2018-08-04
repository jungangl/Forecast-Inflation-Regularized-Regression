function run_BMA(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2, oos_i)
    name = "BMA"
    df_file = "../../data/results/rolling_cv/level$level-h$h-J$J/$name"
    ## Forecast using Bayesian Model Averaging
    θ_vec = 0.05:0.01:0.99
    pre_conv = 1_000
    post_conv = 2_000
    θ, ŷ, RMSE_cv = bma_oos_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, θ_vec, pre_conv, post_conv, oos_i)
    writedlm("$df_file/forecast/$oos_i.csv", ŷ, ',')
    writedlm("$df_file/tuning/$oos_i.csv", θ, ',')
    writedlm("$df_file/RMSE_cv/$oos_i.csv", RMSE_cv, ',')
end



function run_BMA2(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2, oos_i)
    name = "BMA2"
    df_file = "../../data/results/rolling_cv/level$level-h$h-J$J/$name"
    ## Forecast using Bayesian Model Averaging with interaction terms
    θ_vec = 0.05:0.01:0.99
    pre_conv = 1_000
    post_conv = 2_000
    X_lag = hcat(X_lag, X_lag2)
    θ, ŷ, RMSE_cv = bma_oos_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, θ_vec, pre_conv, post_conv, oos_i)
    writedlm("$df_file/forecast/$oos_i.csv", ŷ, ',')
    writedlm("$df_file/tuning/$oos_i.csv", θ, ',')
    writedlm("$df_file/RMSE_cv/$oos_i.csv", RMSE_cv, ',')
end



function run_LAS(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2, oos_i)
    name = "LAS"
    df_file = "../../data/results/rolling_cv/level$level-h$h-J$J/$name"
    #Forecast using a Ridge model with interaction terms
    α = 1.0
    λ_vec = 0.001:0.001:0.75
    β̂, λ, ŷ, RMSE_cv = glmneoos_i_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, λ_vec, α, oos_i)
    writedlm("$df_file/forecast/$oos_i.csv", ŷ, ',')
    writedlm("$df_file/tuning/$oos_i.csv", λ, ',')
    writedlm("$df_file/RMSE_cv/$oos_i.csv", RMSE_cv, ',')
    writedlm("$df_file/beta/$oos_i.csv", β̂, ',')
end



function run_LAS2(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2, oos_i)
    name = "LAS2"
    df_file = "../../data/results/rolling_cv/level$level-h$h-J$J/$name"
    #Forecast using a LASSO model with interaction terms
    α = 1.0
    λ_vec = 0.1:0.01:5.
    X_lag = hcat(X_lag, X_lag2)
    β̂, λ, ŷ, RMSE_cv = glmneoos_i_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, λ_vec, α, oos_i)
    writedlm("$df_file/forecast/$oos_i.csv", ŷ, ',')
    writedlm("$df_file/tuning/$oos_i.csv", λ, ',')
    writedlm("$df_file/RMSE_cv/$oos_i.csv", RMSE_cv, ',')
    writedlm("$df_file/beta/$oos_i.csv", β̂, ',')
end



function run_RDG(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2, oos_i)
    name = "RDG"
    df_file = "../../data/results/rolling_cv/level$level-h$h-J$J/$name"
    #Forecast using a Ridge model with interaction terms
    α = 0.0
    λ_vec = 1.:0.1:25.
    β̂, λ, ŷ, RMSE_cv = glmneoos_i_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, λ_vec, α, oos_i)
    writedlm("$df_file/forecast/$oos_i.csv", ŷ, ',')
    writedlm("$df_file/tuning/$oos_i.csv", λ, ',')
    writedlm("$df_file/RMSE_cv/$oos_i.csv", RMSE_cv, ',')
end



function run_RDG2(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2, oos_i)
    name = "RDG2"
    df_file = "../../data/results/rolling_cv/level$level-h$h-J$J/$name"
    #Forecast using a Ridge model with interaction terms
    α = 0.0
    λ_vec = 1.:0.1:50.
    X_lag = hcat(X_lag, X_lag2)
    β̂, λ, ŷ, RMSE_cv = glmneoos_i_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, λ_vec, α, oos_i)
    writedlm("$df_file/forecast/$oos_i.csv", ŷ, ',')
    writedlm("$df_file/tuning/$oos_i.csv", λ, ',')
    writedlm("$df_file/RMSE_cv/$oos_i.csv", RMSE_cv, ',')
end



function run_RDF(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2, oos_i)
    name = "RDF"
    df_file = "../../data/results/rolling_cv/level$level-h$h-J$J/$name"
    #Forecast using a Random Forrest Regression model
    m_vec = 1:1:25
    n_est = 50
    m, ŷ, RMSE_cv = rfr_oos_forc(J , h, h_lag, J_cv, Y, Y_lag, X_lag, m_vec, n_est, oos_i)
    writedlm("$df_file/forecast/$oos_i.csv", ŷ, ',')
    writedlm("$df_file/tuning/$oos_i.csv", m, ',')
    writedlm("$df_file/RMSE_cv/$oos_i.csv", RMSE_cv, ',')
end



function run_RDF2(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2, oos_i)
    name = "RDF2"
    df_file = "../../data/results/rolling_cv/level$level-h$h-J$J/$name"
    #Forecast using a Random Forrest Regression model
    m_vec = 1:1:25
    n_est = 50
    X_lag = hcat(X_lag, X_lag2)
    m, ŷ, RMSE_cv = rfr_oos_forc(J , h, h_lag, J_cv, Y, Y_lag, X_lag, m_vec, n_est, oos_i)
    writedlm("$df_file/forecast/$oos_i.csv", ŷ, ',')
    writedlm("$df_file/tuning/$oos_i.csv", m, ',')
    writedlm("$df_file/RMSE_cv/$oos_i.csv", RMSE_cv, ',')
end



function dim_1to2(i)
    h_vec = [3, 6, 12]
    oos_i = mod(i, 408)
    h_indx = i ÷ 408
    if oos_i == 0
        oos_i = 408
    else
        h_indx += 1
    end
    h = h_vec[h_indx]
    return oos_i, h
end


## Set the number of cores to use
include("forc-helpers/prepare.jl")
level = 4
h_lag = 12
J = 301
## Different values for different cases
s = ArgParseSettings()
@add_arg_table s begin
    "i"
        arg_type = Int
        required = true
        help = "oos_i"
end
ps = parse_args(s)
i = ps["i"]
oos_i, h = dim_1to2(i)
head_line, Y, Y_lag, X_lag, X_lag2, J_cv = prepare(level, h, J, oos_i)
oos_vec = J - (h_lag + h):length(Y)
oos_num = length(oos_vec)
run_RDG(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2, oos_i)





#=
h = 6
J = 301
name = "short_LAS"
tunes = readdlm("../../data/results/rolling_cv/level4-h$h-J$J/$name/tuning/combined.csv", ',')
ŷ_rolling = readdlm("../../data/results/rolling_cv/level4-h$h-J$J/$name/forecast/combined.csv", ',')
name = "LAS"
ŷ_baseline = convert(Vector{Float64}, readdlm("../../data/results/baseline/level4-h$h-J$J/$name.csv", ',')[2:end])
y_real = convert(Vector{Float64}, readdlm("../../data/results/baseline/level4-h$h-J$J/TRD.csv", ',')[2:end, 1])
sqrt(mean((ŷ_rolling - y_real).^2))
sqrt(mean((ŷ_baseline - y_real).^2))
plot(ŷ_rolling, label = "rolling")
plot!(y_real, lw = 2, label = "")
plot!(ŷ_baseline, label = "baseline")
plot(tunes)
=#
