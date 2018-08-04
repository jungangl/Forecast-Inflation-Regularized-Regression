@everywhere using CSV, GLMNet, Combinatorics, DataFrames, Plots, ArgParse
#@everywhere using ScikitLearn: fit!, predict, @sk_import, fit_transform!
#@everywhere @sk_import ensemble: RandomForestRegressor
include("../forc-helpers/OLSestimator.jl")
include("../forc-helpers/create_matrices.jl")
include("../forc-models/all_models.jl")

function prepare(level, h, J, oos_i)
    ## Choose the head line price to construct the inflation rates
    head_line = true
    J_cv = (J + oos_i + h_lag + h) ÷ 2
    println("h = $h, J = $J, J_cv = $J_cv")
    Y, Y_lag, X_lag, X_lag2 = create_matrices(level, h, h_lag, head_line)
    return head_line, Y, Y_lag, X_lag, X_lag2, J_cv
end
