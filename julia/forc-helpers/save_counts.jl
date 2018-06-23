function save_counts(agg, term)
    levelcounts = zeros(maximum(agg), 2)
    for i in 1:maximum(agg)
        levelcounts[i, :] = [i, sum(level_bools(i, agg, term))]
    end
    levelcounts = convert(Matrix{Int64}, levelcounts)
    writedlm("../../data/source/levelcounts.csv", levelcounts, ',')
end
