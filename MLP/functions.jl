mutable struct dense_layer
    x
    w
    b
    dw
    db
end

mutable struct Sigmoid
    z
end

mutable struct ReLu
    mask
end

mutable struct SoftmaxwithLoss
    y
    t
end

function loss(x, t)
    y = predict(x)
    return cross_entropy_error(y, t)
end

function cross_entropy_error(y,t)
    delta = 1e-7
    batch_size = length(y[:,1])
    return (-sum(log.(y.+delta).*t) / batch_size)
end

function sigmoid(x)
    return 1/(1+exp(-x))
end

function relu(x)
    return max(0,x)
end

function softmax_single(a)
    c = maximum(a)
    exp.(a .- c) / sum(exp.(a .- c))
end

function softmax(a)
    temp = map(softmax_single, eachrow(a))
    return(transpose(hcat(temp ...)))
end


# f는 손실함수, x는 입력값, t는 정답, w는 대상
function numerical_gradient(f, x, t, w)
    h=10^-4
    vec=zeros(Float64,size(w))
    
for i in (1:length(w))
        origin=w[i]
        w[i]+=h
        fx1=f(x,t)
        w[i]-=2*h
        fx2=f(x,t)
        vec[i]=(fx1-fx2)/2h
        w[i]=origin
    end
    return  vec
end

function TwoLayerNet_numerical_gradient(f, x, t)
    grads["W1"] = numerical_gradient(f, x, t,params["W1"])
    grads["W2"] = numerical_gradient(f, x, t,params["W2"])
    grads["b1"] = numerical_gradient(f, x, t,params["b1"])
    grads["b2"] = numerical_gradient(f, x, t,params["b2"])
    return(grads)
end

function evaluate(test_x,test_y)
    temp = (sum((argmax.(eachrow(predict(test_x))).-1) .== test_y)/size(test_x)[1])
    return (temp * 100)
end

function SGD(params,grads)
    for key in keys(params)
        params[key] -= learning_rate * grads[key]
    end
    return params
end

function SoftmaxwithLoss_forward(x,t)
    y = softmax(x)
    loss = cross_entropy_error(y, t)
    result.y = y
    result.t = t
    return loss
end

function SoftmaxwithLoss_backward(result,dout=1)
    batch_size = size(result.t)[1]
    dx = (result.y-result.t) / batch_size
    return dx
end

function dense_layer_forward(dense,x,w,b)
    cal = (x * w) .+ b
    dense.x = x
    dense.w = w
    dense.b = b
    return cal
end

function dense_layer_backward(dense, dout)
    dx = *(dout,Array(dense.w'))
    dense.dw = *(Array(dense.x'), dout)
    dense.db = Array(sum(eachrow(dout))')
    return dx
end

function sigmoid_forward(Sigmoid, x)
    dx = sigmoid.(x)
    Sigmoid.z = dx
    return dx
end

function sigmoid_backward(Sigmoid, dout)
    dx = dout .* (1.0 .- Sigmoid.z) .* Sigmoid.z
    return dx
end

function relu_forward(Relu, x)
    x = relu.(x)
    Relu.mask = x .> 0
    return x
end

function relu_backward(Relu,dout)
    dx = dout.* Relu.mask
    return dx
end
