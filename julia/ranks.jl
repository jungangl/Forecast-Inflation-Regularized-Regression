using CSV
hs = [3, 6, 12]
Js = [301, 373, 493]
models = ["BMA", "BMA2", "LAS", "LAS2", "RDG", "RDG2", "RDF", "RDF2", "ARM", "OLS", "MAG", "MAG2", "DFM", "DFM2", "RWM", "ESMB"]
ensembles = ["BMA", "BMA2", "LAS", "LAS2", "RDG", "RDG2", "RDF", "RDF2", "ARM", "OLS", "MAG", "MAG2", "DFM", "DFM2", "RWM"]
RMSEs = zeros(length(models), length(Js))
for h in hs
    for (j, J) in enumerate(Js)
        println("$h, $J")
        file_path = "../data/result-forc-indi/level4-h$h-J$J/combined.csv"
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
end

df_out = DataFrame()
df_out[:Models] = models
for (j, J) in enumerate(Js)
    df_out[Symbol("Year$J")] = RMSEs[:, j]
end
CSV.write("../data/result-forc-indi/RMSEs.csv", df_out)
