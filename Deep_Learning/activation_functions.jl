# Act.Fuc in hidden layer

function step_function(x)
    if x<=0
        return 0
    else
        return 1
    end
end


function sigmoid(x)
    return 1/(1+exp(-x))
end


function relu(x)
    if x > 0
        return x
    else
        return 0
    end
end


# Act.Fuc in output layer


function identity_function(x)
    return x
end


function softmax_single(a)
    c = maximum(a)
    exp.(a .- c) / sum(exp.(a .- c))
end

# for batch data
function softmax(a)
    temp = map(softmax_single, eachrow(a))
    return(transpose(hcat(temp ...)))
end
