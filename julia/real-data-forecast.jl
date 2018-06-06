Num_Core = 2
if nprocs() != Num_Core
    addprocs(Num_Core - 1)
end
@everywhere using CSV, GLMNet, Combinatorics, DataFrames, Plots
#@everywhere using ScikitLearn: fit!, predict, @sk_import, fit_transform!
#@everywhere @sk_import ensemble: RandomForestRegressor

h = 6 ## horizon (measured in months) ahead we will forecast
lag_h = 12 ## number of months over which to calculate inflation for RHS variables
level = 4 ## aggregation level
CV_J = 250 ## Within sample training set goes from 1 to 250, validation set goes from (250 + h) to 500
J = 500 ## Out of sample goes from (500 + h) onwards

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
PCE_data = data[4:end,:]
agg = convert(Array{Int}, data[2,2:end])
term = convert(Array{Int}, data[3,2:end])
save_counts(agg, term)

agg_bool = level_bools(4, agg, term)
agg_data = PCE_data[:, find([1 agg_bool])]

Headline_PCE = convert(Array{Float64}, PCE_data[:, 2])
Core_PCE = convert(Array{Float64}, PCE_data[:, 3])
agg_PCE = convert(Array{Float64},agg_data[:,2:end])

## Compute h-step ahead headline inflation as in Gamber and Smith (2016)
## Define left hand side variables
hl_π = ((Headline_PCE[lag_h + 1:end] ./ Headline_PCE[1:end - lag_h]) - 1) * (100 / (lag_h / 12))
core_π = ((Core_PCE[lag_h + 1:end] ./ Core_PCE[1:end - lag_h]) - 1) * (100 / (lag_h / 12))
## Define right hand side variables
agg_π = ((agg_PCE[lag_h + 1:end, :] ./ agg_PCE[1:end - lag_h, :]) - 1) .* (100 / (lag_h / 12))

yfull = hl_π
xfull = agg_π



#=
Define generic OLS estimators
=#
function OLSestimator(y, x)
    estimate = inv(x' * x) * (x' * y)
    return estimate
end










#############################################################
#=
Forecast Using OLS
With Only Lagged Aggregate Variable (AR model)
Start with observation from t = 1 to J
Return the forecast values from t = J+1 to T
=#
function AR_oos_forc(J, h, yfull, K = 1)
    T_forc = length((J + h):length(yfull))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(yfull))
        LHS = yfull[(h + K):(t - h)]
        #println("$(h + K):$(t - h)")
        RHS = ones(length(LHS))
        for k in 1:K
            RHS = hcat(RHS, yfull[(1 + K) - k:(t - h) - k - (h - 1)])
            #println("$((1 + K) - k):$((t - h) - k - (h - 1))")
        end
        βhat = OLSestimator(LHS, RHS)
        y_current = [1.0]
        for k in 1:K
            y_current = vcat(y_current, yfull[t - h - (k - 1)])
            #println("$(t - h - (k - 1))")
        end
        ŷ[i] = (βhat' * y_current)[1]
    end
    return ŷ
end

##-------------------------------------------##
K = 1
ŷ = AR_oos_forc(J, h, yfull, K)
RMSE = sqrt(mean((yfull[J + h:end] - ŷ) .^ 2))
println("OLS AR($K) Model, h = $h: \nRMSE = $RMSE")
plot(title = "AR(1) with forecasting $h step ahead")
plot!(yfull[J + h:end], label = "data")
plot!(ŷ, label = "forecast")
savefig("../figures/with-real/AR1-$h-step-ahead.pdf")
##-------------------------------------------##










#############################################################
#=
Forecast Using OLS with Lagged Aggregate Variable
And Lagged Disaggregate Variable
Leaving out the last disaggregate variable
Start with observation from t = 1 to J
Return the forecast values from t = J+1 to T
=#
function OLS_oos_forc(J, h, yfull, xfull)
    T_forc = length((J + h):length(yfull))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(yfull))
        LHS = yfull[1 + h:(t - h)]
        RHS = [ones(length(LHS)) yfull[1:length(LHS)] xfull[1:length(LHS), 1:end - 1]]
        βhat = OLSestimator(LHS, RHS)
        x_crnt = [1.0; yfull[t - h]; xfull[t - h, 1:end - 1]]
        ŷ[i] = (βhat' * x_crnt)[1]
    end
    return ŷ
end

##-------------------------------------------##
ŷ = OLS_oos_forc(J, h, yfull, xfull)
RMSE = sqrt(mean((yfull[J + h:end] - ŷ) .^ 2))
println("OLS Regression Including all Variables, h = $h: \nRMSE = $RMSE")
plot(title = "OLS with forecasting $h step ahead")
plot!(yfull[J + h:end], label = "data")
plot!(ŷ, label = "forecast")
savefig("../figures/with-real/OLS-$h-step-ahead.pdf")
##-------------------------------------------##










#############################################################
#=
Model Average out of sample estimation,
Start with observation from t = 1 to J
Return the forecast values from t = J+1 to T
=#
function modelavg_oos_forc(J, h, Q, yfull, xfull, N = 10)
    T_forc = length((J + h):length(yfull))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(yfull))
        picks = collect(combinations((1:N), Q))
        y_forcs = zeros(length(picks))
        for n in 1:length(picks)
            LHS = yfull[1 + h:(t - h)]
            RHS = [ones(length(LHS)) yfull[1:length(LHS)]]
            RHS = hcat(RHS, xfull[1:length(LHS), picks[n]])
            βhat = OLSestimator(LHS, RHS)
            x_crnt = [1.0; yfull[t - h]; xfull[t - h, picks[n]]]
            y_forcs[n] = (βhat' * x_crnt)[1]
        end
        ŷ[i] = mean(y_forcs)
    end
    return ŷ
end

##-------------------------------------------##
Q = 1
ŷ = modelavg_oos_forc(J, h, Q, yfull, xfull, 10)
RMSE = sqrt(mean((yfull[J + h:end] - ŷ) .^ 2))
println("Equal Weights Model Averaging with Q = $Q, h = $h: \nRMSE = $RMSE")
plot(title = "Model Averaging with forecasting $h step ahead")
plot!(yfull[J + h:end], label = "data")
plot!(ŷ, label = "forecast")
savefig("../figures/with-real/modelavg-$h-step-ahead.pdf")
##-------------------------------------------##










#############################################################
#=
Forecast using a Dynamic Factor Model.
Forecasting model has lagged aggregate
variable and lagged values of first "r"
principal components (factors).
=#

function dfm_oos_forc(J, h, yfull, xfull; r = 1)
    T_forc = length((J + h):length(yfull))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(yfull))
        x_j = xfull[1:t - h,:]
        x_j_c = x_j .- mean(x_j, 1)
        # computing the weight matrix
        covx = cov(x_j_c)
        W = eig(covx)[2]
        W = flipdim(W,2)
        # W is the weight matrix
        factors = (x_j_c * W)[:, 1:r]
        LHS = yfull[1 + h:(t - h)]
        RHS = [ones(length(LHS)) yfull[1:length(LHS)] factors[1:length(LHS), :]]
        βhat = OLSestimator(LHS, RHS)
        #  Compute Forecasts
        x_crnt = [1; yfull[t - h]; factors[t - h, :]]
        ŷ[i] = (βhat' * x_crnt)[1]
    end
    return ŷ
end

##-------------------------------------------##
r = 1
ŷ = dfm_oos_forc(J, h, yfull, xfull; r = r)
RMSE = sqrt(mean((yfull[J + h:end] - ŷ) .^ 2))
println("Dynamic Factor r = $r, h = $h: \nRMSE = $RMSE")
plot(title = "DFM with forecasting $h step ahead")
plot!(yfull[J + h:end], label = "data")
plot!(ŷ, label = "forecast")
savefig("../figures/with-real/DFM-$h-step-ahead.pdf")
##-------------------------------------------##










#############################################################
#=
Forecast using Ridge Regression
with Lagged Aggregate Variable and
Lagged Disaggregate Variables
=#
@everywhere function glmnet_inner_RMSE(J, h, CV_J, yfull, xfull, α, λ)
    ŷ_CV = zeros(length(CV_J + h:J))
    for (j, t) in enumerate(CV_J + h:J)
        LHS = yfull[h + 1:t - h]
        RHS = [yfull[1:length(LHS)] xfull[1:length(LHS), :]]
        res = glmnet(RHS, LHS, alpha = α, lambda = [λ], tol = 10e-20)
        βhat = [res.a0; collect(res.betas)]
        x_crnt = [1;yfull[t - h];xfull[t - h, :]]
        ŷ_CV[j] = (βhat' * x_crnt)[1]
    end
    return [sqrt(mean((yfull[CV_J + h:J] - ŷ_CV) .^ 2)), λ]
end

function glmnet_oos_forc(J, h, CV_J, yfull, xfull, λ_vec, α)
    βhat = []
    # Step1 : pick the best λ using parallel computing
    RMSE_glmnet_CV = zeros(length(λ_vec), 2)
    result = pmap(λ -> glmnet_inner_RMSE(J, h, CV_J, yfull, xfull, α, λ), λ_vec)
    for i in 1:length(λ_vec)
        RMSE_glmnet_CV[i, :] = result[i]
    end
    index = findmin(RMSE_glmnet_CV[:, 1])[2]
    λ = λ_vec[index]
    # Step 2: oos forecasting
    T_forc = length(J + h:length(yfull))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(yfull))
        LHS = yfull[h + 1:t - h]
        RHS = [yfull[1:length(LHS)] xfull[1:length(LHS), :]]
        res = glmnet(RHS, LHS, alpha = α, lambda = [λ], tol = 10e-20)
        βhat = [res.a0; collect(res.betas)]
        x_crnt = [1; yfull[t - h]; xfull[t - h, :]]
        ŷ[i] = (βhat' * x_crnt)[1]
    end
    return βhat, λ, ŷ
end

##-------------------------------------------##
"""
Ridge is a special case of GLMNet when α = 0.0
"""
α_rid = 0.0
λ_vec_rid = 10.:0.01:11
βhat, λ, ŷ = glmnet_oos_forc(J, h, CV_J, yfull, xfull, λ_vec_rid, α_rid)
RMSE = sqrt(mean((yfull[J + h:end] - ŷ) .^ 2))
println("Tuning Parameter Chosen for ridge: $λ")
println("RMSE from ridge Including all Variables, h = $h: \nRMSE = $RMSE")
plot(title = "Ridge with forecasting $h step ahead")
plot!(yfull[J + h:end], label = "data")
plot!(ŷ, label = "forecast")
savefig("../figures/with-real/ridge-$h-step-ahead.pdf")
##-------------------------------------------##


##-------------------------------------------##
"""
LASSO is a special case of GLMNet when α = 1.0
"""
α_las = 1.0
λ_vec_las = 0.4:0.01:0.5
βhat, λ, ŷ = glmnet_oos_forc(J, h, CV_J, yfull, xfull, λ_vec_las, α_las)
RMSE = sqrt(mean((yfull[J + h:end] - ŷ) .^ 2))
println("Tuning Parameter Chosen for lasso: $λ")
println("Percent of Slope Coefficients Set Equal to Zero by lasso: $(round(100 * (sum(βhat[2:end] .==  .0) / length(βhat[2:end])), 2))")
println("RMSE from lasso Including all Variables, h = $h: \nRMSE = $RMSE")
plot(title = "Lasso with forecasting $h step ahead")
plot!(yfull[J + h:end], label = "data")
plot!(ŷ, label = "forecast")
savefig("../figures/with-real/lasso-$h-step-ahead.pdf")
##-------------------------------------------##










#############################################################
#=
Forecast using Random Forest Regression
with Lagged Aggregate Variable and
Lagged Disaggregate Variables

@everywhere function rfr_inner_RMSE(J, h, CV_J, yfull, xfull, n_est, m)
    model = RandomForestRegressor(n_est, max_depth = m, random_state = 0)
    ŷ_CV = zeros(length(CV_J + h:J))
    for (j, t) in enumerate(CV_J + h:J)
        LHS = yfull[h + 1:t - h]
        RHS = [yfull[1:length(LHS)] xfull[1:length(LHS), :]]
        fit!(model, RHS, LHS)
        x_crnt = [yfull[t - h]; xfull[t - h, :]]'
        ŷ_CV[j] = ScikitLearn.predict(model, x_crnt)[1]
    end
    return [sqrt(mean((yfull[CV_J + h:J] - ŷ_CV) .^ 2)), m]
end


function rfr_oos_forc(J , h, CV_J, yfull, xfull, m_vec; n_est = 50)
    # Step1 : pick the best m
    RMSE_rfr_CV = zeros(length(m_vec), 2)
    result = pmap(m -> rfr_inner_RMSE(J, h, CV_J, yfull, xfull, n_est, m), m_vec)
    for i in 1:length(m_vec)
        RMSE_rfr_CV[i,:] = result[i]
    end
    index = findmin(RMSE_rfr_CV[:, 1])[2]
    m = m_vec[index]
    model = RandomForestRegressor(n_est, max_depth = m, random_state = 0)
    # Step 2: oos forecasting
    T_forc = length(J + h:length(yfull))
    ŷ = zeros(T_forc)
    println("Start computing oos RMSE")
    for (i, t) in enumerate((J + h):length(yfull))
        LHS = yfull[h + 1:t - h]
        RHS = [yfull[1:length(LHS)] xfull[1:length(LHS), :]]
        fit!(model, RHS, LHS)
        x_crnt = [yfull[t - h]; xfull[t - h, :]]'
        ŷ[i] = ScikitLearn.predict(model, x_crnt)[1]
    end
    return m, ŷ
end

##-------------------------------------------##
m_vec = 5:1:10
m,ŷ = rfr_oos_forc(J, h, CV_J, yfull, xfull, m_vec)
RMSE = sqrt(mean((yfull[J + h:end] - ŷ) .^ 2))
println("Depth parameter chosen for random Forest: $m")
println("RMSE from Random Forest Including all Variables, h = $h: \nRMSE = $RMSE")
plot(title = "Random Forest Regression with forecasting $h step ahead")
plot!(yfull[J + h:end], label = "data")
plot!(ŷ, label = "forecast")
savefig("../figures/with-real/RFR-$h-step-ahead.pdf")
##-------------------------------------------##
=#









#############################################################
#=
Forecast using Bayesian Model Averaging
=#
@everywhere function MargLik(x_in, y_in, g_in)
    N_f = size(y_in, 1)
    k_f = size(x_in, 2)
    Px = eye(N_f) - x_in * inv(x_in'x_in) * x_in'
    yty = (y_in - mean(y_in))' * (y_in - mean(y_in))
    ymy = y_in'Px * y_in
    g1 = g_in / (g_in + 1)
    g2 = 1 / (g_in + 1)
    ln_fy = .5 * k_f * log.(g1) - .5 * (N_f - 1) * log.(g2 * ymy + g1 * yty)
    return ln_fy[1]
end


@everywhere function bma_forc(g, yfull, xfull, t, h, θ, pre_conv, post_conv)
    LHS = yfull[h + 1:t - h]
    x = [yfull[1:length(LHS)] xfull[1:length(LHS), :]]
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
            x_fore = [yfull[1:(t - h)] xfull[1:(t - h), :]]
            x_fore = x_fore .- mean(x_fore, 1)
            x_fore = [ones(size(x_fore, 1)) x_fore[:, find(τ_vec)]]
            x_crnt = x_fore[end, :]
            y_forc = y_forc + x_crnt' * post_β̄
        end
    end
    return y_forc / (post_conv)
end

@everywhere function bma_inner_RMSE(J, h, CV_J, yfull, xfull, θ, pre_conv, post_conv, g)
    ŷ_CV = zeros(length(CV_J + h:J))
    for (j, t) in enumerate(CV_J + h:J)
        ŷ_CV[j] = bma_forc(g, yfull, xfull, t, h, θ, pre_conv, post_conv)
    end
    return [sqrt(mean((yfull[CV_J + h:J] - ŷ_CV) .^ 2)), g]
end

function bma_oos_forc(J, h, CV_J, yfull, xfull, θ, g_vec; pre_conv = 1000, post_conv = 2000)
    # Step1 : pick the best g
    RMSE_bma_CV = zeros(length(g_vec), 2)
    result = pmap(g -> bma_inner_RMSE(J, h, CV_J, yfull, xfull, θ, pre_conv, post_conv, g), g_vec)
    for i in 1:length(g_vec)
        RMSE_bma_CV[i, :] = result[i]
    end
    index = findmin(RMSE_bma_CV[:, 1])[2]
    g = g_vec[index]
    # Step 2: oos forecasting
    T_forc = length(J + h:length(yfull))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(yfull))
        ŷ[i] = bma_forc(g, yfull, xfull, t, h, θ, pre_conv, post_conv)
    end
    return g, ŷ
end

##-------------------------------------------##
g_vec = 0.0001:0.0001:0.001
θ = 0.5
g,ŷ = bma_oos_forc(J, h, CV_J, yfull, xfull, θ, g_vec)
RMSE = sqrt(mean((yfull[J + h:end] - ŷ) .^ 2))
println("Prior parameter g chosen for BMA: $g")
println("RMSE from Bayesian Model Averaging, h = $h: \nRMSE = $RMSE")
plot(title = "Bayesian Model Averaging with forecasting $h step ahead")
plot!(yfull[J + h:end], label = "data")
plot!(ŷ, label = "forecast")
savefig("../figures/with-real/bayesian-modelavg-$h-step-ahead.pdf")
##-------------------------------------------##
