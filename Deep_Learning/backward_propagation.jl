1. 네트워크 및 초기 매개 변수 설정

"""
TwoLayerNet를 mutable struct로 만듭니다.
"""
mutable struct init_network
    W1
    b1
    W2
    b2
end

function making_network(input_size, hidden_size, output_size, weight_init_std =0.01)
    W1 = weight_init_std * randn(Float64, input_size, hidden_size)
    b1 = zeros(Float32, 1, hidden_size)
    W2 = weight_init_std * randn(Float64, hidden_size, output_size)
    b2 = zeros(Float32, 1, output_size)
    return(init_network(W1, b1, W2, b2))
end

TwoLayerNet = making_network(784, 50, 10)

2. 역전파 인스턴스 설정

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

# 계층마다 인스턴스를 만들어줘야 한다.

result = SoftmaxwithLoss(0,0)
dense1 = dense_layer(0,0,0,0,0)
dense2 = dense_layer(0,0,0,0,0)
Sigmoid1 = Sigmoid(0)
#Relu = ReLu(0)

3. 역전파에 필요한 함수 정의

function cross_entropy_error(y,t)
    delta = 1e-7
    batch_size = length(y[:,1])
    return (-sum(log.(y.+delta).*t) / batch_size)
end

function sigmoid(x)
    return 1/(1+exp(-x))
end

function softmax_single(a)
    c = maximum(a)
    exp.(a .- c) / sum(exp.(a .- c))
end

function softmax(a)
    temp = map(softmax_single, eachrow(a))
    return(transpose(hcat(temp ...)))
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

4. 역전파에 필요한 변수 정의

train_loss_list = Float64[]
accuracy = Float64[]
train_size = size(train_x)[1]
batch_size = 100
learning_rate = 0.1

5. 역전파 알고리즘 코드

@time begin
    for i in 1: 600
        batch_mask = rand(1:train_size, 100)
        x_batch = train_x[batch_mask, :]
        t_batch = t[batch_mask, :]
        
        # 순전파
        z1 = dense_layer_forward(dense1,x_batch,TwoLayerNet.W1,TwoLayerNet.b1)
        a1 = sigmoid_forward(Sigmoid1,z1)
        z2 = dense_layer_forward(dense2,a1,TwoLayerNet.W2,TwoLayerNet.b2)
        num = SoftmaxwithLoss_forward(z2,t_batch)
    
        # 역전파
        last_layer = SoftmaxwithLoss_backward(result)
        z2_back = dense_layer_backward(dense2, last_layer)
        a1_back = sigmoid_backward(Sigmoid1, z2_back)
        z1_back = dense_layer_backward(dense1, a1_back)

        #가중치 갱신
        TwoLayerNet.W1 -= (learning_rate * dense1.dw)
        TwoLayerNet.W2 -= (learning_rate * dense2.dw)
        TwoLayerNet.b1 -= (learning_rate * dense1.db)
        TwoLayerNet.b2 -= (learning_rate * dense2.db)
    
        temp_loss = loss(x_batch, t_batch)
        print("NO.$i: ")
        println(temp_loss)
        append!(train_loss_list, temp_loss)
        append!(accuracy, evaluate(test_x, test_y))
    end
end

6. 손실 함수, 정확도 그래프 그리기

using Plots

# 손실 함수
x = range(1,length(train_loss_list),step=1)
y = train_loss_list
plot(x,y)

# 정확도
x = range(1,length(accuracy),step=1)
y = accuracy
plot(x,y)
