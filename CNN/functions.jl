function relu(x)
    return max(0,x)
end

function softmax(a)
    temp = map(softmax_single, eachrow(a))
    return(transpose(hcat(temp ...)))
end

function softmax_single(a)
    c = maximum(a)
    return exp.(a .- c) / sum(exp.(a .- c))
end

function cross_entropy_error(y,t)
    delta = 1e-7
    batch_size = length(y[:,1])
    return (-sum(log.(y.+delta).*t) / batch_size)
end

function loss_CNN_batch(x, t)
    y = predict(x)
    return cross_entropy_error(y, t)
end

function evaluate_CNN_batch(x, y)
    temp = sum(((argmax.(eachrow(predict(x)))).-1) .== y)/ size(x)[4]
    return (temp * 100)
end

function padding(input, num)
    input_r, input_c, input_d, input_num = size(input)
    temp = zeros(input_r + (2 * num), input_c + (2 * num), input_d, input_num)
    temp_r, temp_c = size(temp)
    temp[(1 + num):(temp_r - num), (1 + num):(temp_c - num), 1: input_d, 1: input_num] = input
    return temp
end

function im2col(input, filter_r,filter_c, stride, pad)
    
    input_r, input_c,input_d,input_num = size(input)
    
    out_r = Int(((input_r + 2*pad - filter_r) ÷ stride) + 1)
    out_c = Int(((input_c + 2*pad - filter_c) ÷ stride) + 1)
    
    img = padding(input, pad)
    col = zeros(out_r, out_c,filter_r,filter_c,input_d,input_num)
    

    for i in 1:filter_r
        r_max = (i + stride *out_r) -1
        for j in 1:filter_c
            c_max = (j + stride *out_c) -1
        
            col[:, :, j, i, :, :] = img[i:stride:r_max, j:stride:c_max,:,:]
        end
    end

    coll = permutedims(col,(4,3,5,2,1,6))
    result = Array(reshape(coll,:,out_r*out_c*input_num)')
    
    #결과는 1324 순으로 도출

    return result
end

function col2im(dense, dcol, filter_r,filter_c, stride, pad)
    
    input_r, input_c,input_d,input_num = size(dense.x)
    
    
    out_r = Int(((input_r + 2*pad - filter_r) ÷ stride) + 1)
    out_c = Int(((input_c + 2*pad - filter_c) ÷ stride) + 1)
    
    temp = Array(dcol')
    mc = reshape(temp,filter_r,filter_c,input_d,out_r,out_c,input_num)
    col = permutedims(mc,(5,4,2,1,3,6))
    
    img = zeros((input_r+2*pad+stride-1),(input_c+2*pad+stride-1), input_d, input_num)
        
    for i in 1:filter_r
        r_max = (i + stride *out_r) -1
        for j in 1:filter_c
            c_max = (j + stride *out_c) -1
        
            img[i:stride:r_max, j:stride:c_max,:,:] += col[:, :, j, i, :, :] 
        end
    end
    
    result = img[1+pad:input_r+pad,1+pad:input_c+pad,:,:]
    
    return result
end


using Random

function drop_out_single(input_size, rate)
    function changing_T_or_F_with_percentage(number, input_size, rate)
        temp_num = input_size * rate
        if number > temp_num
            return 0
        else
            return 1
        end
    end     
    temp = shuffle(reshape(1:input_size, 1, input_size))
    return changing_T_or_F_with_percentage.(temp, input_size, rate)
end

function drop_out(input_size, hidden_size, rate)
    temp = drop_out_single(hidden_size, rate)
    temp_num = input_size - 1
    for i = 1:temp_num
        temp_1 = drop_out_single(hidden_size, rate)
        temp = [temp; temp_1]
    end
    return temp
end

function dropout_forward(dropout, x, dropout_ratio)
    
    if dropout_ratio < 1
        dropout.mask = drop_out(size(x)[1],size(x)[2], dropout_ratio)
        return x .* dropout.mask
        
    else dropout_ratio = 1
        return x .*  (1.0 - dropout_ratio)
    end
end

function dropout_backward(dropout, dout)
    return dout .* dropout.mask
end
