include("MNIST_data.jl")
include("functions.jl")

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
