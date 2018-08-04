###########################################################################################
#Define the marginal likelihood function
###########################################################################################
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



###########################################################################################
#Use MCMCMC method for bayesian model averaging
###########################################################################################
@everywhere function bma_forc(Y, Y_lag, X_lag, vec_î, h, θ, pre_conv, post_conv)
    LHS = Y[1:vec_î - h]
    N = length(LHS)
    x = [Y_lag[1:N] X_lag[1:N, :]]
    x = x .- mean(x, 1)
    K = size(x, 2)
    τ_vec = zeros(Bool, K)
    k = sum(τ_vec)
    RHS = [ones(N) x[:, find(τ_vec)]]
    ln_mod_prior = log(θ ^ k * (1 - θ) ^ (K - k))
    # Compute marginal likelihood for initial model
    g = min(1 / N, 1 / K ^ 2)
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
        println("s = $(s), k = $(k): $(find(τ_vec_new))")
        RHS_new = [ones(N) x[:, find(τ_vec_new)]]
        ln_mod_prior_new = log(θ ^ k * (1 - θ) ^ (K - k))
        # Compute marginal likelihood for candidate model
        ln_fy_new = -Inf
        if k < K
            writedlm("RSM_new.csv", RHS_new, ',')
            writedlm("LHS.csv", LHS, ',')
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
            x_forc = [Y_lag[1:vec_î] X_lag[1:vec_î, :]]
            x_forc = x_forc .- mean(x_forc, 1)
            x_forc = [ones(size(x_forc, 1)) x_forc[:, find(τ_vec)]]
            x_crnt = x_forc[end, :]
            y_forc = y_forc + x_crnt' * post_β̄
        end
    end
    return y_forc / (post_conv)
end



###########################################################################################
#The inner function
###########################################################################################
@everywhere function bma_inner_RMSE(J, h, h_lag, J_cv, Y, Y_lag, X_lag, θ, pre_conv, post_conv, oos_i)
    ## Vector Index for oos forecast goes from J_cv - (h_lag + h) to J - (1 + h_lag + h)
    #println("θ = $θ")
    ŷ_cv = zeros(Float64, size(J_cv - (h_lag + h):J + (oos_i - 1) - (1 + h_lag + h)))
    for (n, vec_î) in enumerate(J_cv - (h_lag + h):J + (oos_i - 1) - (1 + h_lag + h))
        println("n = $n")
        ŷ_cv[n] = bma_forc(Y, Y_lag, X_lag, vec_î, h, θ, pre_conv, post_conv)
    end
    return [sqrt(mean((Y[J_cv - (h_lag + h):J - (1 + h_lag + h)] - ŷ_cv) .^ 2)), θ]
end



###########################################################################################
#The outer function
###########################################################################################
function bma_oos_forc(J, h, h_lag, J_cv, Y, Y_lag, X_lag, θ_vec, pre_conv, post_conv, oos_i)
    # Step1 : pick the best θ
    RMSE_cv = zeros(length(θ_vec), 2)
    result = pmap(θ -> bma_inner_RMSE(J, h, h_lag, J_cv, Y, Y_lag, X_lag, θ, pre_conv, post_conv, oos_i),
                  θ_vec)::Array{Array{Float64,1},1}
    for i in 1:length(θ_vec)
        RMSE_cv[i, :] = result[i]
    end
    index = findmin(RMSE_cv[:, 1])[2]
    θ = θ_vec[index]
    # Step 2: oos forecasting
    ## Vector Index for oos forecast goes from J - (h_lag + h) to length(Y)
    vec_î = (J - (h_lag + h):length(Y))[oos_i]
    ŷ = bma_forc(Y, Y_lag, X_lag, vec_î, h, θ, pre_conv, post_conv)
    return θ, ŷ, RMSE_cv
end
