using DataFrames
using Distributions

#Load full Data Set

data = readtable("PCEPI_DETAIL.csv")
N_all = size(data,1)

PCE_data = data[4:end,:]

agg_category = convert(Array{Int},data[2,2:end])
term_category = convert(Array{Int},data[3,2:end])


agg_2_bool = (agg_category.==2)
agg_3_bool = (agg_category.==3) .| ((agg_category.==2) .& (term_category.==1))
agg_4_bool = (agg_category.==4) .| ((agg_category.==2) .& (term_category.==1)) .| ((agg_category.==3) .& (term_category.==1))

agg_ind_1 = find([1 agg_1_bool])
agg_ind_2 = find([1 agg_2_bool])
agg_ind_3 = find([1 agg_3_bool])
agg_ind_4 = find([1 agg_4_bool])

agg_2_data = PCE_data[:,agg_ind_2]
agg_3_data = PCE_data[:,agg_ind_3]
agg_4_data = PCE_data[:,agg_ind_4]

Headline_PCE = convert(Array{Float64},PCE_data[:,2])
Core_PCE = convert(Array{Float64},PCE_data[:,3])
agg_2_PCE = convert(Array{Float64},agg_2_data[:,2:end])
agg_3_PCE = convert(Array{Float64},agg_3_data[:,2:end])
agg_4_PCE = convert(Array{Float64},agg_4_data[:,2:end])

#Compute k-step ahead headline inflation as in Gamber and Smith (2016)

k = 6 # horizon (number of months) ahead we will forecast
lag_k = 12 # number of months over which to calculate inflation for RHS variables

#Define left hand side variables (either headinline PCE inflation or core PCE inflation)
hl_pie = ((Headline_PCE[lag_k+k+1:end]./Headline_PCE[lag_k+1:end-k])-1)*(100/(k/12))
core_pie = ((Core_PCE[lag_k+k+1:end]./Core_PCE[lag_k+1:end-k])-1)*(100/(k/12))

#Define right hand side variables
hl_pie_lag = ((Headline_PCE[lag_k+1:end-k]./Headline_PCE[1:end-lag_k-k])-1)*(100/(lag_k/12))

agg_2_pie_lag = ((agg_2_PCE[lag_k+1:end-k,:]./agg_2_PCE[1:end-lag_k-k,:])-1).*(100/(lag_k/12))
agg_3_pie_lag = ((agg_3_PCE[lag_k+1:end-k,:]./agg_3_PCE[1:end-lag_k-k,:])-1).*(100/(lag_k/12))
agg_4_pie_lag = ((agg_4_PCE[lag_k+1:end-k,:]./agg_4_PCE[1:end-lag_k-k,:])-1).*(100/(lag_k/12))

# Run linear regression of headline PCE on lagged headline PCE and subcomponent PCE.
# Code below currently uses aggregation level 4

T = size(hl_pie,1)

X = [ones(T) hl_pie_lag agg_4_pie_lag]
Y = hl_pie

betahat = inv(X'X)*X'Y

coef_names = [:Intercept;:Lagged_Aggregate;names(agg_4_data[2:end])]

out_mat = [coef_names betahat]

out_mat_df = convert(DataFrame,out_mat)

rename!(out_mat_df,:x1,:Variable_Name)
rename!(out_mat_df,:x2,:Coefficient_Estimate)

println(out_mat_df)
