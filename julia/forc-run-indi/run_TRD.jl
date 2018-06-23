include("../forc-helpers/prepare_level4_h6_J349.jl")
name = "TRD"
folder_name = "result-forc-indi"
df_file = "../../data/$folder_name/level$level-h$h-J$J/$name.csv"

#=
Save Real Data
=#
df_file = "../../data/$folder_name/level$level-h$h-J$J/$name.csv"
df[:REAL] = Y[J + h:end]
CSV.write(df_file, df)


#=
Forecast Using OLS
With Only Lagged Aggregate Variable (AR model)
Start with observation from t = 1 to J
Return the forecast values from t = J+1 to T
=#
##-------------------------------------------##
ŷ = AR_oos_forc(J, h, Y, Y_lag)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("AR(1) Model: \nRMSE = $RMSE")
df[:ARM] =  ŷ
CSV.write(df_file, df)
##-------------------------------------------##


#=
Forecast Using OLS with Lagged Aggregate Variable
And Lagged Disaggregate Variable
Leaving out the last disaggregate variable
Start with observation from t = 1 to J
Return the forecast values from t = J+1 to T
=#
##-------------------------------------------##
ŷ = OLS_oos_forc(J, h, Y, Y_lag, X_lag)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("OLS Regression Including all Variables: \nRMSE = $RMSE")
df[:OLS] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##


#=
Model Average out of sample estimation,
Start with observation from t = 1 to J
Return the forecast values from t = J+1 to T
=#
##-------------------------------------------##
Q = 1
ŷ = modelavg_oos_forc(J, h, Q, Y, Y_lag, X_lag)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Equal Weights Model Averaging with Q = $Q: \nRMSE = $RMSE")
df[:MAG] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##


#=
Model Average with interactions out of sample estimation,
Start with observation from t = 1 to J
Return the forecast values from t = J+1 to T
=#
##-------------------------------------------##
Q = 1
ŷ = modelavg_oos_forc(J, h, Q, Y, Y_lag, X_lag, X_lag2)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Equal Weights Model Averaging with interactions, Q = $Q: \nRMSE = $RMSE")
df[:MAG2] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##


#=
Forecast using a Dynamic Factor Model.
Forecasting model has lagged aggregate
variable and lagged values of first "r"
principal components (factors).
=#
##-------------------------------------------##
r = 1
ŷ = dfm_oos_forc(J, h, Y, Y_lag, X_lag; r = r)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Dynamic Factor r = $r: \nRMSE = $RMSE")
df[:DFM] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##



#=
Forecast using a Random Walk Model.
It is a special case of AR1 model with AR parameter set to be 1
=#
##-------------------------------------------##
ŷ = rwm_oos_forc(J, h, Y, Y_lag)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Random Walk Model: \nRMSE = $RMSE")
df[:RWM] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##
