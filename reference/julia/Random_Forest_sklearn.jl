using DataFrames
using Plots
train = readtable("/Users/jungangl/Documents/Research/Inflation Forecasting/Code/agg_disagg_data_w_labels.csv")

using ScikitLearn: fit!, predict, @sk_import, fit_transform!

@sk_import ensemble: RandomForestRegressor

function prediction_model(model, predictors)

	J = 100;
	h = 2;

	y = convert(Array, train[:Y])
	y_train = y[h+1:J,1];
	X = convert(Array, train[predictors])
	X_train = X[1:J-h,:];
	X_test = X[J:end-h,:];
	y_test = y[J+h:end,1];

	#Fit the model:

	fit!(model, X_train, y_train)

	#Make predictions on test set
	predictions_test = ScikitLearn.predict(model, X_test)

	#Print out-of-sample RMSE
	RMSE = sqrt(mean((y_test-predictions_test).^2));
	println("\RMSE: ",RMSE)

	return RMSE

end

model = RandomForestRegressor(n_estimators=200, max_depth=2, random_state=0)
predictors = [:Y, :DY1, :DY2, :DY3, :DY4, :DY5, :DY6, :DY7, :DY8, :DY9, :DY10]
prediction_model(model, predictors)

n_vec = collect(1:1:100)
RMSE_vec = zeros(length(n_vec))
for (i,n) in enumerate(n_vec)
    model = RandomForestRegressor(n_estimators=n, max_depth=2, random_state=0)
    RMSE_vec[i] = prediction_model(model, predictors)
    println(i)
end
plot(n_vec,RMSE_vec)

m_vec = collect(1:1:100)
RMSE_vec = zeros(length(m_vec))
for (i,m) in enumerate(m_vec)
    model = RandomForestRegressor(n_estimators=200, max_depth=nothing, random_state=0)
    RMSE_vec[i] = prediction_model(model, predictors)
    println(i)
end
plot(m_vec[1:20],RMSE_vec[1:20])
