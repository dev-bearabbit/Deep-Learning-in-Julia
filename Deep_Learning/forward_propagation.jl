
##### 네트워크 및 초기 매개 변수 설정 ##### 

params = Dict()
grads = Dict()

function making_network(input_size, hidden_size, output_size, weight_init_std =0.01)
    params["W1"] = weight_init_std * randn(Float64, input_size, hidden_size)
    params["b1"] = zeros(Float32, 1, hidden_size)
    params["W2"] = weight_init_std * randn(Float64, hidden_size, output_size)
    params["b2"] = zeros(Float32, 1, output_size)
    return(params)
end

making_network(784, 50, 10)

##### 순전파에 필요한 함수 정의 ##### 

function predict(x)
    a1 = (x * params["W1"]) .+ params["b1"]
    z1 = sigmoid.(a1)
    a2 = (z1 * params["W2"]) .+ params["b2"]
    return softmax(a2)
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

function evaluate(x, t)
    temp = (sum((argmax.(eachrow(predict(test_x))).-1) .== test_y)/size(test_x)[1])
    return (temp * 100)
end

function SGD(params,grads)
    for key in keys(params)
        params[key] -= learning_rate * grads[key]
    end
    return params
end

##### 순전파에 필요한 변수 정의 ##### 

train_size = size(train_x)[1]
batch_size = 100
learning_rate = 0.1
train_loss_list = Float64[]
accuracy = Float64[]
iters_num = 600

##### 순전파 알고리즘 코드 ##### 

@time begin
    for i in 1:iters_num
        batch_mask = rand(1:train_size, 100)
        x_batch = train_x[batch_mask, :]
        t_batch = t[batch_mask, :]
        
        # 편미분값 구하기
        TwoLayerNet_numerical_gradient(loss, x_batch, t_batch)

        # 확률적 경사하강법
        SGD(params, grads)

        temp_loss = loss(x_batch, t_batch)
        print("NO.$i: ")
        println(temp_loss)
        append!(train_loss_list, temp_loss)
        append!(accuracy, evaluate(test_x, test_y))
    end
end

##### 손실 함수, 정확도 그래프 그리기 ##### 

using Plots

# 손실 함수
x = range(1,length(train_loss_list),step=1)
y = train_loss_list
plot(x,y)

# 정확도
x = range(1,length(accuracy),step=1)
y = accuracy
plot(x,y)
