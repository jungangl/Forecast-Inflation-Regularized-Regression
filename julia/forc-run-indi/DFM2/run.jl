## Set the number of cores to use
Num_Core = 1
if nprocs() != Num_Core
    addprocs(Num_Core - 1)
end
include("../../forc-helpers/prepare.jl")
include("../run_all.jl")
level = 4
for h in [3, 6, 12]
    for J in [301, 373, 493]
        h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2 = prepare(level, h, J)
        run_DFM2(h, h_lag, J, J_cv, level, head_line, Y, Y_lag, X_lag, X_lag2)
    end
end
