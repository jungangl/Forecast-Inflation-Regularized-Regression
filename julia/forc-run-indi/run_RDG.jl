include("../forc-helpers/prepare_level4_h6_J277.jl")
name = "RDG"
folder_name = "result-forc-indi"
df_file = "../../data/$folder_name/level$level-h$h-J$J/$name.csv"


#=
Forecast using Ridge Regression
with Lagged Aggregate Variable and
Lagged Disaggregate Variables
Ridge is a special case of GLMNet when α = 0.0
=#
##-------------------------------------------##
α_rid = 0.0
λ_vec_rid = 10:0.1:20
βhat, λ, ŷ = glmnet_oos_forc(J, h, J_cv, Y, Y_lag, X_lag, λ_vec_rid, α_rid)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Tuning Parameter Chosen for ridge: $λ")
println("RMSE from ridge including first order terms: \nRMSE = $RMSE")
df[Symbol(name)] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##
