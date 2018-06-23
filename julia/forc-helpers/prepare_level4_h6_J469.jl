Num_Core = 2
if nprocs() != Num_Core
    addprocs(Num_Core - 1)
end

@everywhere using CSV, GLMNet, Combinatorics, DataFrames, Plots
@everywhere using ScikitLearn: fit!, predict, @sk_import, fit_transform!
@everywhere @sk_import ensemble: RandomForestRegressor

include("../forc-helpers/OLSestimator.jl")
include("../forc-helpers/create_matrices.jl")
include("../forc-models/all_models.jl")


h = 6 ## Horizon (measured in months) ahead we will forecast
h_lag = 12 ## Number of months over which to calculate inflation
J = 469 ## Out of sample goes from (J + h) onwards
J_cv = div(J, 2) ## Within sample training set goes from 1 to J_cv, validation set goes from (J_cv + h) to J
level = 4 ## Aggregation level println("J = $J, J_cv = $J_cv")
head_line = true ## Choose the head line price to construct the inflation rates
println("J = $J, J_cv = $J_cv, h = $h")
df = DataFrame() ## Initialize an empty

Y, Y_lag, X_lag, X_lag2, X_lag3 = create_matrices(level, h, h_lag, head_line)
