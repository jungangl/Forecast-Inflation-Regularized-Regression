Num_Core = 1
if nprocs() != Num_Core
    addprocs(Num_Core - 1)
end


@everywhere using CSV, GLMNet, Combinatorics, DataFrames, Plots
@everywhere using ScikitLearn: fit!, predict, @sk_import, fit_transform!
@everywhere @sk_import ensemble: RandomForestRegressor
include("cross-terms.jl")

h = 6 ## horizon (measured in months) ahead we will forecast
h_lag = 12 ## number of months over which to calculate inflation
level = 4 ## aggregation level
J = 277 ## Out of sample goes from (500 + h) onwards
J_cv = div(J, 2) ## Within sample training set goes from 1 to 250, validation set goes from (250 + h) to 500
println("J = $J, J_cv = $J_cv")
df = DataFrame() ## Initialize an empty
ensemble_names = ["ARM", "MAG", "DFM", "RDG2", "LAS2", "RFM", "BMA"]
df_file = "../data/with-real/actual-forc-data$J.csv"

function OLSestimator(y, x)
    estimate = inv(x' * x) * (x' * y)
    return estimate
end


function level_bools(level, agg, term)
    bools = [false for _ in 1:length(agg)]'
    for i in 1:level - 1
        bools = bools .| ((agg .== i) .& (term .== 1))
    end
    bools = bools .| (agg .== level)
    return bools
end


function save_counts(agg, term)
    levelcounts = zeros(maximum(agg), 2)
    for i in 1:maximum(agg)
        levelcounts[i, :] = [i, sum(level_bools(i, agg, term))]
    end
    levelcounts = convert(Matrix{Int64}, levelcounts)
    writedlm("../data/with-real/levelcounts.csv", levelcounts, ',')
end


## Load full Data Set
data = readtable("../data/with-real/PCEPI_Detail.csv")
PCE = data[4:end,:]
agg = convert(Array{Int}, data[2,2:end])
term = convert(Array{Int}, data[3,2:end])
#save_counts(agg, term)



data_agg = PCE[:, find([1 level_bools(level, agg, term)])]

PCE_hl = convert(Array{Float64}, PCE[:, 2])
PCE_core = convert(Array{Float64}, PCE[:, 3])
PCE_agg = convert(Array{Float64},data_agg[:,2:end])


## Compute h-step ahead headline inflation as in Gamber and Smith (2016)
## Define left hand side variables, divided by (h_lag / 12) to annualize
π_hl = ((PCE_hl[h_lag + h + 1:end] ./ PCE_hl[h_lag + 1:end - h]) - 1) * (100 / (h / 12))
#π_core = ((PCE_core[h_lag + h + 1:end] ./ PCE_core[h_lg + 1:end - h]) - 1) * 100 / (h / 12)
## Define right hand side variables
π_hl_lag = ((PCE_hl[h_lag + 1:end] ./ PCE_hl[1:end - h_lag]) - 1) * (100 / (h_lag / 12))
π_agg_lag = ((PCE_agg[h_lag + 1:end, :] ./ PCE_agg[1:end - h_lag, :]) - 1) * (100 / (h_lag / 12))


Y = π_hl
Y_lag = π_hl_lag
X_lag = π_agg_lag
X_lag2 = second_order_cross(X_lag)
X_lag3 = third_order_cross(X_lag)

#############################################################
#=
Forecast Using OLS
With Only Lagged Aggregate Variable (AR model)
Start with observation from t = 1 to J
Return the forecast values from t = J+1 to T
=#
function AR_oos_forc(J, h, Y, Y_lag)
    T_forc = length((J+h):length(Y))
    ŷ = zeros(T_forc)
    for (i,t) in enumerate((J+h):length(Y))
        LHS = Y[1:(t - h)]
        RHS = [ones(length(LHS)) Y_lag[1:length(LHS)]]
        βhat = OLSestimator(LHS, RHS)
        y_crnt = [1.0; Y_lag[t]]
        ŷ[i] = (βhat' * y_crnt)[1]
    end
    return ŷ
end

##-------------------------------------------##
ŷ = AR_oos_forc(J, h, Y, Y_lag)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("OLS AR(1) Model, h = $h: \nRMSE = $RMSE")
df[:REAL] = Y[J + h:end]
df[:ARM] =  ŷ
##-------------------------------------------##










#############################################################
#=
Forecast Using OLS with Lagged Aggregate Variable
And Lagged Disaggregate Variable
Leaving out the last disaggregate variable
Start with observation from t = 1 to J
Return the forecast values from t = J+1 to T
=#
function OLS_oos_forc(J, h, Y, Y_lag, X_lag)
    T_forc = length((J + h):length(Y))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(Y))
        LHS = Y[1:(t - h)]
        RHS = [ones(length(LHS)) Y_lag[1:length(LHS)] X_lag[1:length(LHS), :]]
        βhat = OLSestimator(LHS, RHS)
        x_crnt = [1.0; Y_lag[t]; X_lag[t, :]]
        ŷ[i] = (βhat' * x_crnt)[1]
    end
    return ŷ
end

##-------------------------------------------##
ŷ = OLS_oos_forc(J, h, Y, Y_lag, X_lag)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("OLS Regression Including all Variables, h = $h: \nRMSE = $RMSE")
df[:OLS] = ŷ
##-------------------------------------------##










#############################################################
#=
Model Average out of sample estimation,
Start with observation from t = 1 to J
Return the forecast values from t = J+1 to T
=#
function modelavg_oos_forc(J, h, Q, Y, Y_lag, X_lag, N = 10)
    T_forc = length((J + h):length(Y))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(Y))
        picks = collect(combinations((1:N), Q))
        y_forcs = zeros(length(picks))
        for n in 1:length(picks)
            LHS = Y[1:(t - h)]
            RHS = [ones(length(LHS)) Y_lag[1:length(LHS)]]
            RHS = hcat(RHS, X_lag[1:length(LHS), picks[n]])
            βhat = OLSestimator(LHS, RHS)
            x_crnt = [1.0; Y_lag[t]; X_lag[t, picks[n]]]
            y_forcs[n] = (βhat' * x_crnt)[1]
        end
        ŷ[i] = mean(y_forcs)
    end
    return ŷ
end

##-------------------------------------------##
Q = 1
ŷ = modelavg_oos_forc(J, h, Q, Y, Y_lag, X_lag, 10)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Equal Weights Model Averaging with Q = $Q, h = $h: \nRMSE = $RMSE")
df[:MAG] = ŷ
##-------------------------------------------##










#############################################################
#=
Forecast using a Dynamic Factor Model.
Forecasting model has lagged aggregate
variable and lagged values of first "r"
principal components (factors).
=#

function dfm_oos_forc(J, h, Y, Y_lag, X_lag; r = 1)
    T_forc = length((J + h):length(Y))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(Y))
        x_j = X_lag[1:t, :]
        x_j_c = x_j .- mean(x_j, 1)
        # computing the weight matrix
        covx = cov(x_j_c)
        W = eig(covx)[2]
        W = flipdim(W,2)
        # W is the weight matrix
        factors = (x_j_c * W)[:, 1:r]
        LHS = Y[1:(t - h)]
        RHS = [ones(length(LHS)) Y_lag[1:length(LHS)] factors[1:length(LHS), :]]
        βhat = OLSestimator(LHS, RHS)
        #  Compute Forecasts
        x_crnt = [1; Y_lag[t]; factors[t, :]]
        ŷ[i] = (βhat' * x_crnt)[1]
    end
    return ŷ
end

##-------------------------------------------##
r = 1
ŷ = dfm_oos_forc(J, h, Y, Y_lag, X_lag; r = r)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Dynamic Factor r = $r, h = $h: \nRMSE = $RMSE")
df[:DFM] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##










#############################################################
#=
Forecast using Ridge Regression
with Lagged Aggregate Variable and
Lagged Disaggregate Variables
=#
@everywhere function glmnet_inner_RMSE(J, h, J_cv, Y, Y_lag, X_lag, α, λ)
    ŷ_cv = zeros(length(J_cv + h:J))
    for (j, t) in enumerate(J_cv + h:J)
        LHS = Y[1:t - h]
        RHS = [Y_lag[1:length(LHS)] X_lag[1:length(LHS), :]]
        res = glmnet(RHS, LHS, alpha = α, lambda = [λ], tol = 10e-20)
        βhat = [res.a0; collect(res.betas)]
        x_crnt = [1; Y_lag[t]; X_lag[t, :]]
        ŷ_cv[j] = (βhat' * x_crnt)[1]
    end
    return [sqrt(mean((Y[J_cv + h:J] - ŷ_cv) .^ 2)), λ]
end

function glmnet_oos_forc(J, h, J_cv, Y, Y_lag, X_lag, λ_vec, α)
    βhat = []
    # Step1 : pick the best λ using parallel computing
    RMSE_glmnet_CV = zeros(length(λ_vec), 2)
    result = pmap(λ -> glmnet_inner_RMSE(J, h, J_cv, Y, Y_lag, X_lag, α, λ), λ_vec)
    for i in 1:length(λ_vec)
        RMSE_glmnet_CV[i, :] = result[i]
    end
    index = findmin(RMSE_glmnet_CV[:, 1])[2]
    λ = λ_vec[index]
    # Step 2: oos forecasting
    T_forc = length(J + h:length(Y))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(Y))
        LHS = Y[1:t - h]
        RHS = [Y_lag[1:length(LHS)] X_lag[1:length(LHS), :]]
        res = glmnet(RHS, LHS, alpha = α, lambda = [λ], tol = 10e-20)
        βhat = [res.a0; collect(res.betas)]
        x_crnt = [1; Y_lag[t]; X_lag[t, :]]
        ŷ[i] = (βhat' * x_crnt)[1]
    end
    return βhat, λ, ŷ
end

##-------------------------------------------##
"""
Ridge is a special case of GLMNet when α = 0.0
"""
α_rid = 0.0
λ_vec_rid = 10:0.1:20
βhat, λ, ŷ = glmnet_oos_forc(J, h, J_cv, Y, Y_lag, X_lag, λ_vec_rid, α_rid)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Tuning Parameter Chosen for ridge: $λ")
println("RMSE from ridge Including all Variables, h = $h: \nRMSE = $RMSE")
df[:RDG] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##


##-------------------------------------------##
"""
Ridge is a special case of GLMNet when α = 0.0
"""
α_rid = 0.0
λ_vec_rid = 10:0.1:20
βhat, λ, ŷ = glmnet_oos_forc(J, h, J_cv, Y, Y_lag, X_lag2, λ_vec_rid, α_rid)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Tuning Parameter Chosen for ridge: $λ, including second order cross terms.")
println("RMSE from ridge Including all Variables, h = $h: \nRMSE = $RMSE")
df[:RDG2] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##


##-------------------------------------------##
"""
Ridge is a special case of GLMNet when α = 0.0
"""
α_rid = 0.0
λ_vec_rid = 10:0.1:20
βhat, λ, ŷ = glmnet_oos_forc(J, h, J_cv, Y, Y_lag, X_lag3, λ_vec_rid, α_rid)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Tuning Parameter Chosen for ridge: $λ, including third order cross terms.")
println("RMSE from ridge Including all Variables, h = $h: \nRMSE = $RMSE")
df[:RDG3] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##
##-------------------------------------------##
##-------------------------------------------##
##-------------------------------------------##
##-------------------------------------------##
##-------------------------------------------##
##-------------------------------------------##
##-------------------------------------------##
"""
LASSO is a special case of GLMNet when α = 1.0
"""
α_las = 1.0
λ_vec_las = 0.1:0.1:2.
βhat, λ, ŷ = glmnet_oos_forc(J, h, J_cv, Y, Y_lag, X_lag, λ_vec_las, α_las)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Tuning Parameter Chosen for lasso: $λ")
println("Percent of Slope Coefficients Set Equal to Zero by lasso: $(round(100 * (sum(βhat[2:end] .==  .0) / length(βhat[2:end])), 2))")
println("RMSE from lasso Including all Variables, h = $h: \nRMSE = $RMSE")
df[:LAS] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##

##-------------------------------------------##
"""
LASSO is a special case of GLMNet when α = 1.0
"""
α_las = 1.0
λ_vec_las = 0.1:0.1:3.
βhat, λ, ŷ = glmnet_oos_forc(J, h, J_cv, Y, Y_lag, X_lag2, λ_vec_las, α_las)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Tuning Parameter Chosen for lasso: $λ, including second order corss terms.")
println("Percent of Slope Coefficients Set Equal to Zero by lasso: $(round(100 * (sum(βhat[2:end] .==  .0) / length(βhat[2:end])), 2))")
println("RMSE from lasso Including all Variables, h = $h: \nRMSE = $RMSE")
df[:LAS2] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##

##-------------------------------------------##
"""
LASSO is a special case of GLMNet when α = 1.0
"""
α_las = 1.0
λ_vec_las = 0.1:0.1:3.
βhat, λ, ŷ = glmnet_oos_forc(J, h, J_cv, Y, Y_lag, X_lag3, λ_vec_las, α_las)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Tuning Parameter Chosen for lasso: $λ, including third order corss terms.")
println("Percent of Slope Coefficients Set Equal to Zero by lasso: $(round(100 * (sum(βhat[2:end] .==  .0) / length(βhat[2:end])), 2))")
println("RMSE from lasso Including all Variables, h = $h: \nRMSE = $RMSE")
df[:LAS3] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##








#############################################################
#=
Forecast using Random Forest Regression
with Lagged Aggregate Variable and
Lagged Disaggregate Variables
=#

@everywhere function rfr_inner_RMSE(J, h, J_cv, Y, Y_lag, X_lag, n_est, m)
    model = RandomForestRegressor(n_est, max_depth = m, random_state = 0)
    ŷ_cv = zeros(length(J_cv + h:J))
    for (j, t) in enumerate(J_cv + h:J)
        LHS = Y[1:t - h]
        RHS = [Y_lag[1:length(LHS)] X_lag[1:length(LHS), :]]
        fit!(model, RHS, LHS)
        x_crnt = [Y_lag[t]; X_lag[t, :]]'
        ŷ_cv[j] = ScikitLearn.predict(model, x_crnt)[1]
    end
    return [sqrt(mean((Y[J_cv + h:J] - ŷ_cv) .^ 2)), m]
end


function rfr_oos_forc(J , h, J_cv, Y, Y_lag, X_lag, m_vec; n_est = 50)
    # Step1 : pick the best m
    RMSE_rfr_CV = zeros(length(m_vec), 2)
    result = pmap(m -> rfr_inner_RMSE(J, h, J_cv, Y, Y_lag, X_lag, n_est, m), m_vec)
    for i in 1:length(m_vec)
        RMSE_rfr_CV[i,:] = result[i]
    end
    index = findmin(RMSE_rfr_CV[:, 1])[2]
    m = m_vec[index]
    model = RandomForestRegressor(n_est, max_depth = m, random_state = 0)
    # Step 2: oos forecasting
    T_forc = length(J + h:length(Y))
    ŷ = zeros(T_forc)
    println("Start computing oos RMSE")
    for (i, t) in enumerate((J + h):length(Y))
        LHS = Y[1:t - h]
        RHS = [Y_lag[1:length(LHS)] X_lag[1:length(LHS), :]]
        fit!(model, RHS, LHS)
        x_crnt = [Y_lag[t]; X_lag[t, :]]'
        ŷ[i] = ScikitLearn.predict(model, x_crnt)[1]
    end
    return m, ŷ
end

##-------------------------------------------##
m_vec = 1:1:25
m,ŷ = rfr_oos_forc(J , h, J_cv, Y, Y_lag, X_lag, m_vec)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Depth parameter chosen for random Forest: $m")
println("RMSE from Random Forest Including all Variables, h = $h: \nRMSE = $RMSE")
df[:RFR] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##










#############################################################
#=
Forecast using Bayesian Model Averaging
=#
@everywhere function MargLik(x, y, g)
    N = size(y, 1)
    k = size(x, 2)
    P = eye(N) - x * inv(x'x) * x'
    yty = (y - mean(y))' * (y - mean(y))
    ymy = y'P * y
    g1 = g / (g + 1)
    g2 = 1 / (g + 1)
    ln_fy = .5 * k * log.(g1) - .5 * (N - 1) * log.(g2 * ymy + g1 * yty)
    return ln_fy[1]
end


@everywhere function bma_forc(g, Y, Y_lag, X_lag, t, h, θ, pre_conv, post_conv)
    LHS = Y[1:t - h]
    x = [Y_lag[1:length(LHS)] X_lag[1:length(LHS), :]]
    x = x .- mean(x, 1)
    N = length(LHS)
    K = size(x, 2)
    τ_vec = zeros(Bool, K)
    k = sum(τ_vec)
    RHS = [ones(N) x[:, find(τ_vec)]]
    ln_mod_prior = log(θ ^ k * (1 - θ) ^ (K - k))
    # Compute marginal likelihood for initial model
    ln_fy = MargLik(RHS, LHS, g)
    y_forc = 0.
    # Begin MC3 simulations
    for s = 1:(pre_conv + post_conv)
        # Generate candidate Neighborhood Model
        neighbor_indx = rand(1:(1 + K))
        # Specify candidate model
        τ_vec_new = deepcopy(τ_vec)
        if neighbor_indx >= 2
            τ_vec_new[neighbor_indx - 1] = !τ_vec[neighbor_indx - 1]
        end
        k = sum(τ_vec_new)
        RHS_new = [ones(N) x[:, find(τ_vec_new)]]
        ln_mod_prior_new = log(θ ^ k * (1 - θ) ^ (K - k))
        # Compute marginal likelihood for candidate model
        ln_fy_new = -Inf
        if k < K
            ln_fy_new = MargLik(RHS_new, LHS, g)
        end
        # MH Step
        prob_acc = min(exp(ln_fy_new + ln_mod_prior_new - (ln_fy + ln_mod_prior)), 1)
        if rand() <= prob_acc
            ln_fy = ln_fy_new
            τ_vec = deepcopy(τ_vec_new)
        end
        # If post-convergence, form forecasts
        if s > pre_conv
            x_selected = x[:, find(τ_vec)]
            post_β̄ = [mean(LHS); inv((1 + g) * (x_selected' * x_selected)) * (x_selected' * LHS)]
            x_fore = [Y_lag[1:t] X_lag[1:t, :]]
            x_fore = x_fore .- mean(x_fore, 1)
            x_fore = [ones(size(x_fore, 1)) x_fore[:, find(τ_vec)]]
            x_crnt = x_fore[end, :]
            y_forc = y_forc + x_crnt' * post_β̄
        end
    end
    return y_forc / (post_conv)
end

@everywhere function bma_inner_RMSE(J, h, J_cv, Y, Y_lag, X_lag, θ, pre_conv, post_conv, g)
    ŷ_cv = zeros(length(J_cv + h:J))
    for (j, t) in enumerate(J_cv + h:J)
        ŷ_cv[j] = bma_forc(g, Y, Y_lag, X_lag, t, h, θ, pre_conv, post_conv)
    end
    return [sqrt(mean((Y[J_cv + h:J] - ŷ_cv) .^ 2)), g]
end

function bma_oos_forc(J, h, J_cv, Y, Y_lag, X_lag, θ, g_vec; pre_conv = 1000, post_conv = 2000)
    # Step1 : pick the best g
    RMSE_bma_CV = zeros(length(g_vec), 2)
    result = pmap(g -> bma_inner_RMSE(J, h, J_cv, Y, Y_lag, X_lag, θ, pre_conv, post_conv, g), g_vec)
    for i in 1:length(g_vec)
        RMSE_bma_CV[i, :] = result[i]
    end
    index = findmin(RMSE_bma_CV[:, 1])[2]
    g = g_vec[index]
    # Step 2: oos forecasting
    T_forc = length(J + h:length(Y))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(Y))
        ŷ[i] = bma_forc(g, Y, Y_lag, X_lag, t, h, θ, pre_conv, post_conv)
    end
    return g, ŷ
end

##-------------------------------------------##
g_vec = 0.0001:0.0001:0.0001
θ = 0.5
g, ŷ = bma_oos_forc(J, h, J_cv, Y, Y_lag, X_lag, θ, g_vec)
RMSE = sqrt(mean((Y[J + h:end] - ŷ) .^ 2))
println("Prior parameter g chosen for BMA: $g")
println("RMSE from Bayesian Model Averaging, h = $h: \nRMSE = $RMSE")
df[:BMA] = ŷ
CSV.write(df_file, df)
##-------------------------------------------##





#############################################################
#=
Forecast using Ensemble
=#
df = CSV.read(df_file)
N = size(df, 1)
K = length(ensemble_names)
ensemble_ŷ = zeros(N)
for name in ensemble_names
    ensemble_ŷ = ensemble_ŷ + convert(Vector{Float64}, df[Symbol(name)]) / K
end
RMSE = sqrt(mean((Y[J + h:end] - ensemble_ŷ) .^ 2))
println("Ensemble from $ensemble_names,")
println("RMSE from Ensemble: RMSE = $RMSE")
df[:ESMB] = ensemble_ŷ
CSV.write(df_file, df)
