function making_network(W, b, weight_size, output_shape, weight_init)
    
    params = Dict()
    
    if weight_init == "std"
        for i in (1:length(W))
            params[W[i]] = 0.01 * randn(Float64, weight_size[i])
            params[b[i]] = zeros(Float64,1, weight_size[i][end])
        end
        return(params)
        
    elseif weight_init == "Xavier"
        for i in (1:length(W))
            params[W[i]] = ((1.0 / prod(output_shape[i]))^(1/2)) * randn(Float64, weight_size[i])
            params[b[i]] = zeros(Float64,1, weight_size[i][end])
        end
        return(params)
        
    elseif weight_init == "He"
        for i in (1:length(W))
            params[W[i]] = ((2.0 /prod(output_shape[i]))^(1/2)) * randn(Float64, weight_size[i])
            params[b[i]] = zeros(Float64,1, weight_size[i][end])
        end
        return(params)
    end
    return(params)
end
end
