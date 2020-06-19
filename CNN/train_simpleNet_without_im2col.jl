include("MNIST_data.jl")
include("functions.jl")
include("layers_without_im2col.jl")
include("making_network.jl")
include("optimizers.jl")

# 손실함수 위한 예측함수 설정
function  predict(input)
    
    pconv_1 = conv2D_forward_batch(pre_dense, input, params["W1"], params["b1"],1)
    pconv_Re = relu.(pconv_1)
    ppool_1 = maxpooling2D_forward_batch(pre_pool,pconv_Re, 2, 2)
    flatten_1 = flatten_forward_batch(pre_flatten,ppool_1)
    dense_1 = (flatten_1 * params["W2"]) .+ params["b2"]
    dense_relu = relu.(dense_1)
    dense_2 = (dense_relu * params["W3"]) .+ params["b3"]
    result = softmax(dense_2)
end

# 가중치 및 편향 설정
W =["W1","W2","W3"]
b = ["b1","b2","b3"]
weight_size = [(5,5,1,30),(4320,100),(100,10)];

params = making_network(W, b, weight_size, (28,28,1,100),"std");

# predict() 파라미터 맞추기 위한 저장소 설정
pre_dense = dense_layer(0,0,0,0,0,0)
pre_pool= repository(0,0,0)
pre_flatten = repository(0,0,0)

# 역전파 위한 저장소 설정
result = SoftmaxwithLoss(0,0)
dense1 = dense_layer(0,0,0,0,0,0)
dense2 = dense_layer(0,0,0,0,0,0)
dense3 = dense_layer(0,0,0,0,0,0)
Relu1 = repository(0,0,0)
Relu2 = repository(0,0,0)
optimizer = optimizers(0,0,0,0)
pool1= repository(0,0,0)
flatten1 = repository(0,0,0)

grads = Dict()
train_loss_list= []

# 학습 모델 (1에폭 기준, 미니 배치 적용)

@time begin
    for i in 1:600
    
        batch_size = rand(1:size(train_x)[4],100)
        train_x_batch = train_x[:,:,:,batch_size]
        t_batch = reshape(t[batch_size,:],100,10)

        #신경망 계산
        conv_1 = conv2D_forward_batch(dense1,train_x_batch,params["W1"],params["b1"],1)
        conv_Re = relu_forward(Relu1, conv_1)
        pool_1 = maxpooling2D_forward_batch(pool1, conv_Re, 2, 2)
        flatten_1 = flatten_forward_batch(flatten1, pool_1)
        dense_1 = dense_layer_forward(dense2,flatten_1,params["W2"],params["b2"])
        dense_relu = relu_forward(Relu2, dense_1)
        dense_2 = dense_layer_forward(dense3,dense_relu,params["W3"],params["b3"])
        num = SoftmaxwithLoss_forward(dense_2,t_batch)

        #역전파 알고리즘
        last_layer = SoftmaxwithLoss_backward(result)
        dense_2_back = dense_layer_backward(dense3, last_layer)
        grads["W3"] = dense3.dw
        grads["b3"] = dense3.db
        dense_relu_back = relu_backward(Relu2, dense_2_back)
        dense_1_back = dense_layer_backward(dense2, dense_relu_back)
        grads["W2"] = dense2.dw
        grads["b2"] = dense2.db
        flatten_1_back = flatten_backward_batch(flatten1,dense_1_back)
        pool_1_back = maxpooling2D_backward_batch(pool1, flatten_1_back, 2, 2)
        conv_Re_back = relu_backward(Relu1, pool_1_back)
        conv_back = conv2D_backward_batch(dense1,conv_Re_back)
        grads["W1"] = dense1.dw
        grads["b1"] = dense1.db
    
        #가중치 갱신
        Adam(params,grads)
    
        temp_loss = loss_CNN_batch(train_x_batch,t_batch)
        print("NO.$i: ")
        println(temp_loss)
        append!(train_loss_list, temp_loss)
    end
end

# 테스트셋으로 정확도 확인 
evaluate_CNN_batch(test_x, test_y)
