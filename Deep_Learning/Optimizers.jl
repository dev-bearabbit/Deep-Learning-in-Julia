function SGD(params,grads)
    for key in keys(params)
        params[key] -= learning_rate * grads[key]
    end
    return params
end

function Momentum(params,grads)
    momentum = 0.9
    if optimizer.v == 0
        optimizer.v = Dict()
        for key in keys(params)
            optimizer.v[key] = zeros(size(params[key]))
            
        end
    end
    
    for key in keys(params)
        optimizer.v[key] = (optimizer.v[key].* momentum) - (learning_rate * grads[key])
        params[key] += optimizer.v[key]
    end
    return params
end

function AdaGrad(params,grads)
    if optimizer.h == 0
        optimizer.h = Dict()
        for key in keys(params)
            optimizer.h[key] = zeros(size(params[key]))
        end
    end
    
    for key in keys(params)
        optimizer.h[key] +=  grads[key] .* grads[key]
        params[key] -= (learning_rate * grads[key]) ./ (optimizer.h[key].^(1/2).+ 1e-7)
    end
    return params
end

function Adam(params,grads,learning_rate = 0.001)
    
    beta1 = 0.9
    beta2 = 0.999
    
    if optimizer.m == 0
        optimizer.m = Dict()
        optimizer.v = Dict()
        for key in keys(params)
            optimizer.m[key] = zeros(size(params[key]))
            optimizer.v[key] = zeros(size(params[key]))
        end
    end
    
    optimizer.iter += 1
    lr_t = learning_rate * (1.0 - beta2^(optimizer.iter))^(1/2) / (1.0 - beta1^(optimizer.iter))
    
    for key in keys(params)
        optimizer.m[key] += (1 - beta1) * (grads[key] - optimizer.m[key])
        optimizer.v[key] += (1 - beta2) * (grads[key].^2 - optimizer.v[key])
        params[key] -= (lr_t * optimizer.m[key]) ./ ((optimizer.v[key]).^(1/2) .+ 1e-7)
    end
    return params
end
