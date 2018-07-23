using CSV, DataFrames
HS = [3, 6, 12]
Js = [301, 373, 493]
ensembles = ["BMA", "BMA2", "LAS", "LAS2", "RDG", "RDG2", "RDF", "RDF2", "ARM", "OLS", "MAG", "MAG2", "DFM", "DFM2", "RWM"]
models = vcat(ensembles, "ESMB")
RMSEs = zeros(length(models), length(HS), length(Js))
for (h, H) in enumerate(HS)
    for (j, J) in enumerate(Js)
        println("$h, $J")
        file_path = "../data/result-forc-indi/level4-h$H-J$J/combined.csv"
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
            RMSEs[m, h, j] = sqrt(mean((ŷ .- y).^2))
        end
    end
end


for (h, H) in enumerate(HS)
    df_out = DataFrame()
    df_out[:Models] = models
    for (j, J) in enumerate(Js)
        df_out[Symbol("h$H J$J")] = RMSEs[:, h, j]
        CSV.write("../data/result-forc-indi/RMSEs_h$H.csv", df_out)
    end
end
