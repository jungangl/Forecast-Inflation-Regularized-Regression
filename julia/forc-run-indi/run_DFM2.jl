include("../forc-helpers/prepare_level4_h6_J277.jl")

name = "DFM2"
folder_name = "result-forc-indi"
df_file = "../../data/$folder_name/level$level-h$h-J$J/$name.csv"

#=
Forecast using a Dynamic Factor Model.
Forecasting model has lagged aggregate variable
and lagged values of first 1 principal components
for the first order terms and second order interactions terms each.
=#
##-------------------------------------------##
r1 = 1
r2 = 1
ŷ = dfm_oos_forc(J, h, Y, Y_lag, X_lag, X_lag2; r1 = r1, r2 = r2)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Dynamic Factor with interaction terms r1 = $(r1), r2 = $(r2): \nRMSE = $RMSE")
df[Symbol(name)] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##
