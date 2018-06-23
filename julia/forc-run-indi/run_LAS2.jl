include("../forc-helpers/prepare_level4_h6_J277.jl")

name = "LAS2"
folder_name = "result-forc-indi"
df_file = "../../data/$folder_name/level$level-h$h-J$J/$name.csv"


#=
Forecast using LASSO with second order cross terms
with Lagged Aggregate Variable and
Lagged Disaggregate Variables
LASSO is a special case of GLMNet when α = 1.0
=#
##-------------------------------------------##
α_las = 1.0
λ_vec_las = 0.1:0.1:5.
βhat, λ, ŷ = glmnet_oos_forc(J, h, J_cv, Y, Y_lag, X_lag2, λ_vec_las, α_las) RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Tuning Parameter Chosen for lasso: $λ, including second order terms.")
println("Percent of Slope Coefficients Set Equal to Zero by lasso: $(round(100 * (sum(βhat[2:end] .==  .0) / length(βhat[2:end])), 2))")
println("RMSE from lasso Including all Variables, h = $h: \nRMSE = $RMSE")
df[Symbol(name)] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##
