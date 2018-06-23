using CSV
Js = [277, 349, 469]
models = ["BMA", "LAS", "LAS2", "LAS3", "RDG", "RDG2", "RFM", "ARM", "OLS", "MAG", "DFM", "ESMB"]
ensembles = ["LAS2", "RDG2", "ARM", "DFM"]
RMSEs = zeros(length(models), length(Js))
for (j, J) in enumerate(Js)
    file_path = "../data/result-forc-indi/level4-h6-J$J/combined.csv"
    df_in = CSV.read(file_path)
    y = convert(Vector{Float64}, df_in[Symbol("REAL")])
    y_e = zeros(length(y))
    for e in ensembles
        y_e = y_e .+ convert(Vector{Float64}, df_in[Symbol(e)]) ./ length(ensembles)
    end
    df_in[:ESMB] = y_e
    CSV.write(file_path, df_in)
    for (m ,model) in enumerate(models)
        ŷ = convert(Vector{Float64}, df_in[Symbol(model)])
        RMSEs[m, j] = sqrt(mean((ŷ .- y).^2))
    end
end
df_out = DataFrame()
df_out[:Models] = models
for (j, J) in enumerate(Js)
    df_out[Symbol("Year$J")] = RMSEs[:, j]
end
CSV.write("../data/result-forc-indi/RMSEs.csv", df_out)
