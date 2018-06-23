include("../forc-helpers/prepare_level4_h6_J277.jl")


name = "RFM2"
folder_name = "result-forc-indi"
df_file = "../../data/$folder_name/level$level-h$h-J$J/$name.csv"


## Forecast using Random Forest Regression
## with Lagged Aggregate Variable and
## Lagged Disaggregate Variables
##-------------------------------------------##
m_vec = 1:1:25
m,ŷ = rfr_oos_forc(J , h, J_cv, Y, Y_lag, X_lag, X_lag2, m_vec)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Depth parameter chosen for Random Forest with interaction terms: $m")
println("RMSE from Random Forest Including all Variables, h = $h: \nRMSE = $RMSE")
df[Symbol(name)] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##
