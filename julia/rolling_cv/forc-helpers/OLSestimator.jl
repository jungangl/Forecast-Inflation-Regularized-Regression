function OLSestimator(y, x)
    estimate = inv(x' * x) * (x' * y)
    return estimate
end
