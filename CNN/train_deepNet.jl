# 손실함수 위한 예측 레이어
function  predict(input)
    
    pconv_1 = convolution2D_forward(pre_dense, input, params["W1"], params["b1"],1,1)
    pconv_Re_1 = relu.(pconv_1)
    pconv_2 = convolution2D_forward(pre_dense, pconv_Re_1, params["W2"], params["b2"],1,1)
    pconv_Re_2 = relu.(pconv_2)    
    ppool_1 = Maxpooling_forward(pre_pool,pconv_Re_2, 2, 2, 2, 0)
    
    pconv_3 = convolution2D_forward(pre_dense, ppool_1, params["W3"], params["b3"],1,1)
    pconv_Re_3 = relu.(pconv_3)
    pconv_4 = convolution2D_forward(pre_dense, pconv_Re_3, params["W4"], params["b4"],1,2)
    pconv_Re_4 = relu.(pconv_4)    
    ppool_2 = Maxpooling_forward(pre_pool,pconv_Re_4, 2, 2, 2, 0)    

    pconv_5 = convolution2D_forward(pre_dense, ppool_2, params["W5"], params["b5"],1,1)
    pconv_Re_5 = relu.(pconv_5)
    pconv_6 = convolution2D_forward(pre_dense, pconv_Re_5, params["W6"], params["b6"],1,1)
    pconv_Re_6 = relu.(pconv_6)    
    ppool_3 = Maxpooling_forward(pre_pool,pconv_Re_6, 2, 2, 2, 0) 
    
    flatten_1 = flatten_forward_batch(pre_flatten, ppool_3)
    dense_1 = (flatten_1 * params["W7"]) .+ params["b7"]
    dense_relu = relu.(dense_1)
    pdropout_1 = dropout_forward(pre_dropout, dense_relu, 0.5)
    dense_2 = (dense_relu * params["W8"]) .+ params["b8"]
    pdropout_2 = dropout_forward(pre_dropout, dense_2, 0.5)
    result = softmax(dense_2)
end

# 가중치, 편향 설정
W =["W1","W2","W3","W4","W5","W6","W7","W8"]
b = ["b1","b2","b3","b4","b5","b6","b7","b8"]
weight_size = [(3,3,1,16),(3,3,16,16),(3,3,16,32),(3,3,32,32),(3,3,32,64),(3,3,64,64),(1024,50),(50,10)];

params = making_network(W, b, weight_size, (28,28,1,100),"std");

# predict() 위한 설정
pre_dense = dense_layer(0,0,0,0,0,0)
pre_pool= repository(0,0,0)
pre_flatten = repository(0,0,0)
pre_dropout = repository(0,0,0)


# 중간값 저장소 for 역전파
result = SoftmaxwithLoss(0,0)
dense1 = dense_layer(0,0,0,0,0,0)
dense2 = dense_layer(0,0,0,0,0,0)
dense3 = dense_layer(0,0,0,0,0,0)
dense4 = dense_layer(0,0,0,0,0,0)
dense5 = dense_layer(0,0,0,0,0,0)
dense6 = dense_layer(0,0,0,0,0,0)
dense7 = dense_layer(0,0,0,0,0,0)
dense8 = dense_layer(0,0,0,0,0,0)
Relu1 = repository(0,0,0)
Relu2 = repository(0,0,0)
Relu3 = repository(0,0,0)
Relu4 = repository(0,0,0)
Relu5 = repository(0,0,0)
Relu6 = repository(0,0,0)
Relu7 = repository(0,0,0)
optimizer = optimizers(0,0,0,0)
pool1 = repository(0,0,0)
pool2 = repository(0,0,0)
pool3 = repository(0,0,0)
flatten1 = repository(0,0,0)
dropout1 = repository(0,0,0)
dropout2 = repository(0,0,0)

# 미분값, 손실값 저장
grads = Dict()
train_loss_list= []

# 학습 모델
@time begin
    for i in 1:600
    
        batch_size = rand(1:size(train_x)[4],100)
        train_x_batch = train_x[:,:,:,batch_size]
        t_batch = reshape(t[batch_size,:],100,10)
        
        #신경망 계산
        conv_1 = convolution2D_forward(dense1,train_x_batch,params["W1"],params["b1"],1,1)
        conv_Re = relu_forward(Relu1, conv_1)
        conv_2 = convolution2D_forward(dense2,conv_Re,params["W2"],params["b2"],1,1)
        conv_Re_2 = relu_forward(Relu2, conv_2)     
        pool_1 = Maxpooling_forward(pool1, conv_Re_2, 2, 2, 2, 0)
        
        conv_3 = convolution2D_forward(dense3,pool_1,params["W3"],params["b3"],1,1)
        conv_Re_3 = relu_forward(Relu3, conv_3)
        conv_4 = convolution2D_forward(dense4,conv_Re_3,params["W4"],params["b4"],1,2)
        conv_Re_4 = relu_forward(Relu4, conv_4)     
        pool_2 = Maxpooling_forward(pool2, conv_Re_4, 2, 2, 2, 0)        
        
        conv_5 = convolution2D_forward(dense5,pool_2,params["W5"],params["b5"],1,1)
        conv_Re_5 = relu_forward(Relu5, conv_5)
        conv_6 = convolution2D_forward(dense6,conv_Re_5,params["W6"],params["b6"],1,1)
        conv_Re_6 = relu_forward(Relu6, conv_6)     
        pool_3 = Maxpooling_forward(pool3, conv_Re_6, 2, 2, 2, 0)
        
        flatten_1 = flatten_forward_batch(flatten1, pool_3)
        dense_1 = dense_layer_forward(dense7,flatten_1,params["W7"],params["b7"])
        dense_relu = relu_forward(Relu7, dense_1)
        #dropout_1 = dropout_forward(dropout1, dense_relu, 0.5)
        dense_2 = dense_layer_forward(dense8,dense_relu,params["W8"],params["b8"])
        #dropout_2 = dropout_forward(dropout2, dense_2, 0.5)
        num = SoftmaxwithLoss_forward(dense_2,t_batch)

        #역전파 알고리즘
        last_layer = SoftmaxwithLoss_backward(result)
        #dropout_2_back = dropout_backward(dropout2, last_layer)
        dense_2_back = dense_layer_backward(dense8, last_layer)
        grads["W8"] = dense8.dw
        grads["b8"] = dense8.db
        #dropout_1_back = dropout_backward(dropout1, dense_2_back)
        dense_relu_back = relu_backward(Relu7,dense_2_back)
        dense_1_back = dense_layer_backward(dense7, dense_relu_back)
        grads["W7"] = dense7.dw
        grads["b7"] = dense7.db
        flatten_1_back = flatten_backward_batch(flatten1,dense_1_back)
        
        pool_3_back = Maxpooling_backward(pool3, flatten_1_back, 2, 2, 2, 0)
        conv_Re_6_back = relu_backward(Relu6, pool_3_back)
        conv_6_back = convolution2D_backward(dense6,conv_Re_6_back,1,1)
        grads["W6"] = dense6.dw
        grads["b6"] = dense6.db
        conv_Re_5_back = relu_backward(Relu5, conv_6_back)
        conv_5_back = convolution2D_backward(dense5,conv_Re_5_back ,1,1)
        grads["W5"] = dense5.dw
        grads["b5"] = dense5.db
        
        pool2_back = Maxpooling_backward(pool2, conv_5_back, 2, 2, 2, 0)
        conv_Re_4_back = relu_backward(Relu4, pool2_back)
        conv_4_back = convolution2D_backward(dense4,conv_Re_4_back,1,2)
        grads["W4"] = dense4.dw
        grads["b4"] = dense4.db
        conv_Re_3_back = relu_backward(Relu3, conv_4_back)
        conv_3_back = convolution2D_backward(dense3,conv_Re_3_back ,1,1)
        grads["W3"] = dense3.dw
        grads["b3"] = dense3.db
        
        pool1_back = Maxpooling_backward(pool1, conv_3_back, 2, 2, 2, 0)
        conv_Re_2_back = relu_backward(Relu2, pool1_back)
        conv_2_back = convolution2D_backward(dense2,conv_Re_2_back,1,1)
        grads["W2"] = dense2.dw
        grads["b2"] = dense2.db
        conv_Re_1_back = relu_backward(Relu1, conv_2_back)
        conv_1_back = convolution2D_backward(dense1,conv_Re_1_back ,1,1)
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

# 테스트셋 정확도
evaluate_CNN_batch(test_x, test_y)
