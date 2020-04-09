function mean_squared_error(y,t)
    error=(y-t).^2
    return sum(error)/length(y)
end


function cross_entropy_error(y,t)
    delta=1e-7
    return -sum(log.(y.+delta).*t)
end
