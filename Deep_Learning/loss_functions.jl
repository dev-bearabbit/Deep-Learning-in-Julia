# 평균 제곱 오차
function mean_squared_error(y,t)
    error=(y-t).^2
    return sum(error)/length(y)
end

# 교차 엔트로피 오차 (single)
function cross_entropy_error(y,t)
    delta=1e-7
    return -sum(log.(y.+delta).*t)
end

# 교차 엔트로피 오차 (batch)

function cross_entropy_error(y,t)
    delta = 1e-7
    batch_size = length(y[:,1])
    return (-sum(log.(y.+delta).*t)/batch_size)
end
