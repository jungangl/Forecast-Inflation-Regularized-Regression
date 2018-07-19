function second_order_cross(X)
    n = size(X, 1)
    k = size(X, 2)
    extra_k = div((1 + k) * k ,2)
    X_2 = zeros(n, extra_k)
    count = 0
    for i in 1:k
        for j in 1:k
            if i <= j
                count = count + 1
                X_2[:, count] = X[:, i] .* X[:, j]
            end
        end
    end
    return X_2
end

function third_order_cross(X)
    n = size(X, 1)
    k = size(X, 2)
    extra_k = div(k * (1 + k) * (2 + k),  6)
    X_3 = zeros(n, extra_k)
    count = 0
    for i in 1:k
        for j in 1:k
            for z in 1:k
                if (i <= j) && (j <= z)
                    count = count + 1
                    X_3[:, count] = X[:, i] .* X[:, j] .* X[:, z]
                end
            end
        end
    end
    return X_3
end
