#############################################################

## Forecast using Bayesian Model Averaging
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


@everywhere function bma_forc(Y, Y_lag, X_lag, t, h, θ, pre_conv, post_conv)
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

@everywhere function bma_inner_RMSE(J, h, J_cv, Y, Y_lag, X_lag, θ, pre_conv, post_conv)
    ŷ_cv = zeros(length(J_cv + h:J))
    for (j, t) in enumerate(J_cv + h:J)
        #println("t = $t")
        ŷ_cv[j] = bma_forc(Y, Y_lag, X_lag, t, h, θ, pre_conv, post_conv)
    end
    return [sqrt(mean((Y[J_cv + h:J] - ŷ_cv) .^ 2)), θ]
end

function bma_oos_forc(J, h, J_cv, Y, Y_lag, X_lag, θ_vec; pre_conv = 1_000, post_conv = 2_000)
    # Step1 : pick the best θ
    RMSE_bma_CV = zeros(length(θ_vec), 2)
    result = pmap(θ -> bma_inner_RMSE(J, h, J_cv, Y, Y_lag, X_lag, θ, pre_conv, post_conv), θ_vec)
    for i in 1:length(θ_vec)
        RMSE_bma_CV[i, :] = result[i]
    end
    index = findmin(RMSE_bma_CV[:, 1])[2]
    θ = θ_vec[index]
    # Step 2: oos forecasting
    T_forc = length(J + h:length(Y))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(Y))
        ŷ[i] = bma_forc(Y, Y_lag, X_lag, t, h, θ, pre_conv, post_conv)
    end
    return θ, ŷ
end



## Forecast using Bayesian Model Averaging with interaction terms
function bma_oos_forc(J, h, J_cv, Y, Y_lag, X_lag, X_lag2, θ_vec; pre_conv = 10_000, post_conv = 20_000)
    X_lag = [X_lag X_lag2]
    # Step1 : pick the best θ
    RMSE_bma_CV = zeros(length(θ_vec), 2)
    result = pmap(θ -> bma_inner_RMSE(J, h, J_cv, Y, Y_lag, X_lag, θ, pre_conv, post_conv), θ_vec)
    for i in 1:length(θ_vec)
        RMSE_bma_CV[i, :] = result[i]
    end
    index = findmin(RMSE_bma_CV[:, 1])[2]
    θ = θ_vec[index]
    # Step 2: oos forecasting
    T_forc = length(J + h:length(Y))
    ŷ = zeros(T_forc)
    for (i, t) in enumerate((J + h):length(Y))
        ŷ[i] = bma_forc(Y, Y_lag, X_lag, t, h, θ, pre_conv, post_conv)
    end
    return θ, ŷ
end
