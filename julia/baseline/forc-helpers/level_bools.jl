function level_bools(level, agg, term)
    bools = [false for _ in 1:length(agg)]'
    for i in 1:level - 1
        bools = bools .| ((agg .== i) .& (term .== 1))
    end
    bools = bools .| (agg .== level)
    return bools
end
