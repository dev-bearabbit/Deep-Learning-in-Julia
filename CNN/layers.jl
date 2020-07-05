# 구조체 정의
mutable struct dense_layer
    x
    w
    col
    col_w
    dw
    db
end

mutable struct repository
    x
    count
    mask
end

mutable struct SoftmaxwithLoss
    y
    t
end

function Maxpooling_forward(pool, input, pool_h, pool_w, stride, pad)
    
    input_r, input_c,input_d,input_num = size(input)
    
    out_r = Int(1 + (input_r +2*pad - pool_h) ÷ stride)
    out_c = Int(1 + (input_c +2*pad - pool_w) ÷ stride)

    ## 1234 순서로 인덱스를 뽑아야 한다.
    col_ex = im2col(input, pool_h, pool_w, stride, pad)
    col= zeros(size(col_ex));
    order = reshape(Vector(1:size(col_ex)[2]),pool_w,pool_h,input_d)
    count = []
    
    for i in 1:size(order)[3]
        temp = reshape(Array(order[:,:,i]'),1,:)
        append!(count, temp)
    end
    
    for i in 1:size(col_ex)[2]
        col[:,count[i]] = col_ex[:,i]
    end
    
    col = Array(col') #1234
    coll = reshape(col, pool_h * pool_w, out_r*out_c*input_num*input_d)
    arg_max = argmax(coll, dims = 1)
    result = maximum(coll,dims = 1)
    out = reshape(result,input_d,out_c,out_r,input_num)
    out = permutedims(out,(3,2,1,4))
    
    pool.x = input
    pool.count = count
    pool.mask = arg_max
    
    return out
end

function Maxpooling_backward(pool,dout,pool_h,pool_w,stride,pad)
    
    dout = permutedims(dout,(3,2,1,4))
    pool_size = pool_h * pool_w
    

    dmax = zeros(pool_size,length(dout))
    dmax_ex = zeros(pool_size,length(dout))
    dmax[reshape(pool.mask,length(dout),1)] = reshape(dout,length(dout),1)
    
    # 1234에서 1324순으로 변환
    for i in 1:size(dmax)[1]
        dmax_ex[i,:] = dmax[pool.count[i],:]
    end
    
    dmax_ex = Array(dmax_ex')
    dx = col2im(pool, dmax_ex, pool_h, pool_w, stride, pad)
    
    return dx
end

function convolution2D_forward(dense ,input, filter, bias, stride, pad)
    
    input_r, input_c,input_d,input_num = size(input)
    filter_r, filter_c, filter_d, filter_num = size(filter)
    
    out_r = Int(((input_r + 2*pad - filter_r) ÷ stride) + 1)
    out_c = Int(((input_c + 2*pad - filter_c) ÷ stride) + 1)
    
    col = im2col(input, filter_r, filter_c, stride, pad)
    col_w = reshape(filter,filter_r*filter_c*filter_d,filter_num)
    
    out = col * col_w .+ bias
    temp = Array(out')
    temp2 = reshape(temp,filter_num,out_r,out_c,input_num)
    result = permutedims(temp2,(3,2,1,4))
    
    dense.x = input
    dense.w = filter
    dense.col = col
    dense.col_w = col_w
    
    return result
end

function convolution2D_backward(dense ,input, stride, pad)

    input_r, input_c,input_d,input_num = size(dense.x)
    filter_r, filter_c, filter_d, filter_num = size(dense.w) 
    
    result = permutedims(input,(3,2,1,4))
    dout = Array(reshape(result,filter_num,:)')
    
    dense.db = sum(dout,dims=1)
    weight = Array(dense.col') * dout
    dense.dw = reshape(weight,size(dense.w))

    # 1324 순서
    dcol = dout * Array(dense.col_w')
    dx = col2im(dense, dcol, filter_r, filter_c, stride, pad)
    
    return dx
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
    return cal
end

function dense_layer_backward(dense, dout)
    dx = *(dout,Array(dense.w'))
    dense.dw = *(Array(dense.x'), dout)
    dense.db = Array(sum(eachrow(dout))')
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

function flatten_forward_batch(re, input)
    
    input_r, input_c, input_d, input_num = size(input)
    out = zeros(input_num, size(input)[1]*size(input)[2]*size(input)[3])
    re.mask = size(input)
    
    for k in 1:input_num
        
        data = input[:,:,:,k]
        temp = reshape(data,1,input_r*input_c*input_d)
        out[k,:] = temp
    end
    
    return out
end

function flatten_backward_batch(re, input)
    
    input_num, input_total = size(input)
    out = zeros(re.mask)
    
    for k in 1:input_num
        dx = reshape(input[k,:],re.mask[1:3])
        out[:,:,:,k] = dx
    end
    return out
end
