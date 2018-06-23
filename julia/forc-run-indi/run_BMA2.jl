include("../forc-helpers/prepare_level4_h6_J277.jl")
name = "BMA2"
folder_name = "result-forc-indi"
df_file = "../../data/$folder_name/level$level-h$h-J$J/$name.csv"

#=
Forecast using Bayesian Model Averaging with interaction terms
=#
##-------------------------------------------##
θ_vec = 0.01:0.01:0.99
θ, ŷ = bma_oos_forc(J, h, J_cv, Y, Y_lag, X_lag, θ_vec)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Prior parameter θ chosen for $(name) with interactions: $θ")
println("RMSE from Bayesian Model Averaging, h = $h: \nRMSE = $RMSE")
df[Symbol(name)] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##
