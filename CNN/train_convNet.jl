include("MNIST_data.jl")
include("functions.jl")
include("layers.jl")
include("making_network.jl")
include("optimizers.jl")

# 로스값용 함수 (레이어 설정 필요)
function  predict(input)
    
    pconv_1 = convolution2D_forward(pre_dense, input, params["W1"], params["b1"],1,0)
    pconv_Re_1 = relu.(pconv_1)
    ppool_1 = Maxpooling_forward(pre_pool,pconv_Re_1, 2, 2, 2, 0)
    
    pconv_2 = convolution2D_forward(pre_dense,ppool_1, params["W2"], params["b2"],1,0)
    pconv_Re_2 = relu.(pconv_2)    
    ppool_2 = Maxpooling_forward(pre_pool,pconv_Re_2, 2, 2, 2, 0)
    
    pconv_3 = convolution2D_forward(pre_dense, ppool_2, params["W3"], params["b3"],1,0)
    pconv_Re_3 = relu.(pconv_3)   

    flatten_1 = flatten_forward_batch(pre_flatten, pconv_Re_3)
    dense_1 = (flatten_1 * params["W4"]) .+ params["b4"]
    dense_relu = relu.(dense_1)
    dense_2 = (dense_relu * params["W5"]) .+ params["b5"]
    result = softmax(dense_2)
end

# 가중치, 편향 생성
W =["W1","W2","W3","W4","W5"]
b = ["b1","b2","b3","b4","b5"]
weight_size = [(3,3,1,32),(3,3,32,64),(3,3,64,64),(576,64),(64,10)];
output_shape = [(28,28,1),(13,13,32),(5,5,64),(1,576),(1,64)]

params = making_network(W, b, weight_size,output_shape,"He");

# predict용 저장소(사용x)
pre_dense = dense_layer(0,0,0,0,0,0)
pre_pool= repository(0,0,0)
pre_flatten = repository(0,0,0)

# 실제 저장소
result = SoftmaxwithLoss(0,0)
dense1 = dense_layer(0,0,0,0,0,0)
dense2 = dense_layer(0,0,0,0,0,0)
dense3 = dense_layer(0,0,0,0,0,0)
dense4 = dense_layer(0,0,0,0,0,0)
dense5 = dense_layer(0,0,0,0,0,0)
Relu1 = repository(0,0,0)
Relu2 = repository(0,0,0)
Relu3 = repository(0,0,0)
Relu4 = repository(0,0,0)
optimizer = optimizers(0,0,0,0)
pool1 = repository(0,0,0)
pool2 = repository(0,0,0)
flatten1 = repository(0,0,0)

# 미분값, 손실값 저장
grads = Dict()
train_loss_list= []

## 학습 모델 (convNet)
# batch_size = 100 / 1_epoch = 600

@time begin
    for i in 1:600
    
        batch_size = rand(1:size(train_x)[4],100)
        train_x_batch = train_x[:,:,:,batch_size]
        t_batch = t[batch_size,:]
        
        #신경망 계산
        conv_1 = convolution2D_forward(dense1,train_x_batch,params["W1"],params["b1"],1,0)
        conv_Re = relu_forward(Relu1, conv_1)
        pool_1 = Maxpooling_forward(pool1, conv_Re, 2, 2, 2, 0)
        
        conv_2 = convolution2D_forward(dense2,pool_1,params["W2"],params["b2"],1,0)
        conv_Re_2 = relu_forward(Relu2, conv_2)     
        pool_2 = Maxpooling_forward(pool2, conv_Re_2, 2, 2, 2, 0)
        
        conv_3 = convolution2D_forward(dense3,pool_2,params["W3"],params["b3"],1,0)
        conv_Re_3 = relu_forward(Relu3, conv_3)

        flatten_1 = flatten_forward_batch(flatten1, conv_Re_3)
        dense_1 = dense_layer_forward(dense4,flatten_1,params["W4"],params["b4"])
        dense_relu = relu_forward(Relu4, dense_1)
        dense_2 = dense_layer_forward(dense5,dense_relu ,params["W5"],params["b5"])
        num = SoftmaxwithLoss_forward(dense_2 ,t_batch)

        #역전파 알고리즘
        last_layer = SoftmaxwithLoss_backward(result)
        dense_2_back = dense_layer_backward(dense5,last_layer)
        grads["W5"] = dense5.dw
        grads["b5"] = dense5.db
        dense_relu_back = relu_backward(Relu4,dense_2_back)
        dense_1_back = dense_layer_backward(dense4, dense_relu_back)
        grads["W4"] = dense4.dw
        grads["b4"] = dense4.db
        flatten_1_back = flatten_backward_batch(flatten1,dense_1_back)
        
        conv_Re_3_back = relu_backward(Relu3, flatten_1_back)
        conv_3_back = convolution2D_backward(dense3,conv_Re_3_back ,1,0)
        grads["W3"] = dense3.dw
        grads["b3"] = dense3.db
        
        pool2_back = Maxpooling_backward(pool2, conv_3_back , 2, 2, 2, 0)
        conv_Re_2_back = relu_backward(Relu2, pool2_back)
        conv_2_back = convolution2D_backward(dense2,conv_Re_2_back,1,0)
        grads["W2"] = dense2.dw
        grads["b2"] = dense2.db
        
        pool1_back = Maxpooling_backward(pool1, conv_2_back, 2, 2, 2, 0)
        conv_Re_1_back = relu_backward(Relu1, pool1_back)
        conv_1_back = convolution2D_backward(dense1,conv_Re_1_back ,1,0)
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

# 테스트셋 정확도 계산
evaluate_CNN_batch(test_x, test_y)
    
