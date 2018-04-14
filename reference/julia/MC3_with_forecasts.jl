using DataFrames
using Distributions
using Plots

function MargLik(x_in, y_in, g_in)
    N_f = size(y_in,1)
    k_f = size(x_in,2)
    Px = eye(N_f) - x_in*inv(x_in'x_in)*x_in'
    yty = (y_in-mean(y_in))'*(y_in-mean(y_in))
    ymy = y_in'Px*y_in
    g1 = g_in/(g_in+1)
    g2 = 1/(g_in+1)
    ln_fy = .5*k_f*log.(g1) - .5*(N_f-1)*log.(g2*ymy + g1*yty)
    return ln_fy
end


#Load full Data Set
data = readtable("agg_disagg_data_w_labels.csv")
N_all = size(data,1)


# Define Relevant Parameters and Storage Spaces for Out of Sample Forecasting Exercise
h = 2  # Forecast horizon
J = 100 # Last value of LHS data used in first training sample is period J. First out of sample forecast is for period J+h
y_fore_vec = [];


# Define fixed variables
num_sim =  2000  # Total Number of simulations
num_conv = 1000  # Number of pre-convergence simulations


# Specify prior over models. All model priors assume that each individual regressor is included in the true model with
# the same prior probability theta, and this probabiity is independent of the inclusion of other regressors. Thus, the probability of a model is:
# Pr(M_i) = theta^k_i*(1-theta)^(k_total-k_i)
# where k_i is the number of regressors included in model i
# 1 = fixed theta priors as in Fernandez, Ley and Steel (2001). Theta is chosen as a prior parameter.
# FLS (2001) choose theta = 0.5, which makes all models equally likely.
# 2 = random theta hierarchical prior as in Ley and Steel (2009). A beta(a,b) prior is placed on theta.
# a is set to 1, and b is calibrated to a particular choice for the mean prior model size. */

model_prior = 1;
if model_prior == 1;
    theta=0.5;
end;

if model_prior == 2;
    a=1;
    prior_mean_model_size = 999;
    #If prior_mean_model_size is set to 999, it will be reset to half of the possible number of covariates (k_total) below (when k_total is defined)
end;


# Begin Out of Sample Forecasting Exercise
for oos_indic = J:1:(N_all-h)
    y = convert(Array{Float64}, data[h+1:oos_indic,[:Y]])
    x = convert(Array{Float64}, data[1:oos_indic-h,:])
    N = size(y,1)
    x = x .- mean(x,1)
    k_total = size(x,2)
    if model_prior==2
        if prior_mean_model_size==999
            prior_mean_model_size = k_total/2
        end
        b = (k_total-prior_mean_model_size)/prior_mean_model_size
    end

    # Initialize storage matrices for MC3 algorithm
    y_fore = 0
    # Specify g for g-prior. This uses the recommendation of Fernandez, Ley and Steele (2001, Journal of Econometrics)
    g = 1/max(k_total^2,N)
    # Specify Initial Model
    x_indic_old = zeros(k_total)
    x_old = [ones(N) x[:,find(x_indic_old)]]
    k = size(x_old,2)-1
    if k==k_total
        println("Initial model can't have all variables - X matrix is not of full rank")
        println("Program terminating")
        quit()
    end;


    # Model prior for initial model
    if model_prior==1;
        ln_model_prior_old = log(theta^k*(1-theta)^(k_total-k));
    elseif model_prior==2;
        ln_model_prior_old = log((gamma(a+b)/(gamma(a)*gamma(b)))*((gamma(a+k)*gamma(b+k_total-k))/gamma(a+b+k_total)));
    end;


    # Compute marginal likelihood for initial model
    ln_fy_old = MargLik(x_old, y, g)[1]
    # Begin MC3 simulations
    for sim = 1:num_sim
        # Print Progress
        if mod(sim,10000)==0
            println("Simulation Number: $sim")
        end
        # Determine number of neighborhood models
        num_mod_current = 1  # current model
        num_mod_diff = k_total

        # Generate candidate neighborhood model
        num_mod = num_mod_current + num_mod_diff
        gen_c = ceil(rand()*num_mod)
        gen_c = convert(Int,gen_c)

        # Specify candidate model
        if gen_c == 1
            x_indic_new = copy(x_indic_old)
        else
            gen_c = gen_c - num_mod_current
            if x_indic_old[gen_c] == 1
                x_indic_new = copy(x_indic_old)
                x_indic_new[gen_c] = 0
            else
                x_indic_new = copy(x_indic_old)
                x_indic_new[gen_c] = 1
            end
        end

        x_new = [ones(N) x[:,find(x_indic_new)]]
        k = size(x_new,2)-1
        # Model prior for candidate model
        if model_prior==1;
            ln_model_prior_new = log((theta^k)*((1-theta)^(k_total-k)));
        elseif model_prior==2;
            ln_model_prior_new = log((gamma(a+b)/(gamma(a)*gamma(b)))*((gamma(a+k)*gamma(b+k_total-k))/gamma(a+b+k_total)));
        end;

        # Compute marginal likelihood for candidate model
        if k<k_total
            ln_fy_new = MargLik(x_new, y, g)
        elseif k==k_total
            ln_fy_new = log(0)
        end


        # MH Step
        prob_acc = min(exp(ln_fy_new+ln_model_prior_new-(ln_fy_old+ln_model_prior_old)),1)
        u = rand()
        if u <= prob_acc[]
            ln_fy_old = copy(ln_fy_new)
            x_indic_old = copy(x_indic_new)
        end


        # If post-convergence, form forecasts
        if sim > num_conv
            x_old = x[:,find(x_indic_old)]
            Post_mean_B = inv((1+g)*(x_old'*x_old))*(x_old'*y)
            Post_mean_B = [mean(y);Post_mean_B]
            x_fore = convert(Array, data[1:oos_indic,1:end])
            x_fore = x_fore .- mean(x_fore,1)
            x_fore = [ones(size(x_fore,1)) x_fore[:,find(x_indic_old)]]
            x_fore = x_fore[end,1:end]
            y_fore = y_fore + x_fore'*Post_mean_B
        end

    end

    y_fore = y_fore/(num_sim-num_conv)[1];
    y_fore_vec = push!(y_fore_vec,y_fore)

end

# Ouput Results

y_actual = convert(Array{Float64}, data[J+h:N_all,1])

forecast_mat = [y_fore_vec y_actual]
forecast_mat_df = convert(DataFrame,forecast_mat)
println(forecast_mat_df)

SE = (y_actual.-y_fore_vec).^2
MSE = mean(SE)
RMSE = sqrt(MSE)

println("h: $h")
println("RMSE: $RMSE")
