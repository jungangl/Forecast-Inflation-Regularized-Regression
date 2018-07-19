include("cross_terms.jl")
include("level_bools.jl")



function create_matrices(level, h, h_lag, head_line)
    ## Load full Data Set
    data = readtable("../../../data/source/PCEPI_Detail.csv")::DataFrame
    PCE = data[4:end, :]::DataFrame
    agg = convert(Array{Int}, data[2, 2:end])
    term = convert(Array{Int}, data[3, 2:end])
    data_agg = PCE[:, find([1 level_bools(level, agg, term)])]

    ## Compute h-step ahead headline inflation as in Gamber and Smith (2016)
    ## Define left-hand-side lagged variables, divided by (h_lag / 12) to annualize
    PCE_hl = convert(Array{Float64}, PCE[:, [2]])
    π_hl = ((PCE_hl[h_lag + h + 1:end] ./ PCE_hl[h_lag + 1:end - h]) - 1) * (100 / (h / 12))
    ## Define right-hand-side lagged variables
    π_hl_lag = ((PCE_hl[h_lag + 1:end] ./ PCE_hl[1:end - h_lag]) - 1) * (100 / (h_lag / 12))

    ## Compute h-step ahead headline inflation as in Gamber and Smith (2016)
    ## Define left hand-side-variables, divided by (h_lag / 12) to annualize
    PCE_core = convert(Array{Float64}, PCE[:, [3]])
    π_core = ((PCE_core[h_lag + h + 1:end] ./ PCE_core[h_lag + 1:end - h]) - 1) * 100 / (h / 12)
    ## Define right hand side lagged variables
    π_core_lag = ((PCE_core[h_lag + 1:end] ./ PCE_core[1:end - h_lag]) - 1) * (100 / (h_lag / 12))

    ## Define the right-hand-side disaggregate variables
    PCE_agg = convert(Array{Float64}, data_agg[:, 2:end])
    π_agg_lag = ((PCE_agg[h_lag + 1:end, :] ./ PCE_agg[1:end - h_lag, :]) - 1) * (100 / (h_lag / 12))

    Y = head_line .* π_hl + (1 - head_line) .* π_core
    Y_lag = head_line .* π_hl_lag + (1 - head_line) .* π_core_lag
    X_lag = π_agg_lag
    X_lag2 = second_order_cross(X_lag)

    return Y, Y_lag, X_lag, X_lag2
end
