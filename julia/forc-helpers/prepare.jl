@everywhere using CSV, GLMNet, Combinatorics, DataFrames, Plots
#@everywhere using ScikitLearn: fit!, predict, @sk_import, fit_transform!
#@everywhere @sk_import ensemble: RandomForestRegressor

include("../forc-helpers/OLSestimator.jl")
include("../forc-helpers/create_matrices.jl")
include("../forc-models/all_models.jl")

function prepare(level, h, J)
    h_lag = 12 ## Number of months over which to calculate inflation
    J_cv = (J - (1 + h_lag + h)) ÷ 2 + (1 + h_lag + h) ## Half of the within sample size used as for cross validation
    head_line = true ## Choose the head line price to construct the inflation rates
    println("J = $J, J_cv = $J_cv, h = $h")
    Y, Y_lag, X_lag, X_lag2 = create_matrices(level, h, h_lag, head_line)
    return h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2
end
