# convolution layers

function conv_2d_forward(input, filter, stride)
    
    input_r, input_c = size(input)
    filter_r, filter_c = size(filter)
    
    if (((input_r-filter_r) * (1/stride))+1) % 1 != 0
        print("error")
    elseif (((input_c-filter_c) * (1/stride))+1) % 1 != 0
        print("error")
    end
    
    result = zeros(Int(((input_r-filter_r) * (1/stride))+1), Int(((input_r-filter_r) * (1/stride))+1))
    
    for i in 1:size(result)[1]
        for j in 1:size(result)[1]
            result[i,j] = sum(input[1+stride*(i-1):1+filter_r-1+stride*(i-1), 1+stride*(j-1):1+filter_c-1+stride*(j-1)] .* filter)
        end
    end
    return result
end

function making_index(input, filter)

    input_r, input_c = size(input)
    filter_r, filter_c = size(filter)
    num = ((input_r -1)*1) + filter_r
    
    # 간격만들기
    first = fill(1,filter_r-1)
    last = fill(input_r,filter_r-1)
    middle = collect(1:input_r)
    first_num = append!(first, middle)
    second_num = append!(middle, last)
    result = []

    for i in 1:num

            temp = (first_num[i]:second_num[i])
            push!(result,temp)
    end
    
    return result
end

function conv_2d_backward(input, filter)
    
    input_r, input_c = size(input)
    filter_r, filter_c = size(filter)
    
    num = ((input_r -1)*1) + filter_r
    out = zeros(num,num)
    
    input_index = making_index(input, filter)
    filter_index =  making_index(filter, filter)
    
    for i in 1:(length(input_index)-length(filter_index))
        temp = (1:filter_r)
        push!(filter_index, temp)
    end

    filter_index = sort!(filter_index)

    for j in 1:num
        for k in 1:num

            ix = reshape(input[input_index[j],input_index[k]],1,size(input[input_index[j],input_index[k]])[1] * size(input[input_index[j],input_index[k]])[2])
            ex = reverse(reshape(filter[filter_index[j],filter_index[k]],size(filter[filter_index[j],filter_index[k]])[1] * size(filter[filter_index[j],filter_index[k]])[2]))
            to = (ix * ex)[1]
            out[j,k] = to
        end
    end
    
    return out
end

function conv2D_forward_batch(dense,input, W, b, stride)
    
    input_r, input_c, input_d, input_num = size(input)
    filter_r, filter_c, filter_d, filter_num = size(W)
    num = Int(((input_r-filter_r) * (1/stride))+1)
    result =[]
    total = zeros(num,num,filter_num,input_num)
    dense.w = W
    dense.x = input
    
    if (((input_r-filter_r) * (1/stride))+1) % 1 != 0
        print("error")
    elseif (((input_r-filter_r) * (1/stride))+1) % 1 != 0
        print("error")
    elseif input_d != filter_d
        print("error")
    end
    
    for k in 1:input_num
        
        data = input[:,:,:,k]

        if input_d == 1
            for j in 1:filter_num
                temp = conv_2d_forward(data[:,:,1], W[:,:,:,j], stride) .+ b[j]
            
                total[:,:,j,k] = temp
            end
    
        elseif input_d > 1
            for j in 1:filter_num
                for i in 1:input_d
                    temp = conv_2d_forward(data[:,:,i], W[:,:,i,j], stride)
                    push!(result, temp)
                end

                temp = zeros(size(result[1]))

                for i in 1:input_d
                    temp = temp .+ result[i]
                end
                
                temp = temp .+ b[j]
                total[:,:,j,k] = temp
                result = []
            end
        end
    end
    
    return total
end

# 스트라이드가 1인 경우만 가능한 역전파
        
function conv2D_backward_batch(dense,dout)
    
    input_r, input_c, input_d, input_num = size(dout)
    filter_r, filter_c, filter_d, filter_num = size(dense.w)
    out_r, out_c, out_d = size(dense.x)
    dx = zeros(size(dense.x))
    dw = zeros(size(dense.w))
    db = zeros(1,input_d)
    total = zeros(size(dw)[1:2])
    bias = 0
    
    
    # 가중치, 편향 미분
    for j in 1:input_d
        for i in 1:out_d
            for k in 1:input_num

                temp = conv_2d_forward(dense.x[:,:,i,k],dout[:,:,j,k],1)
                total += temp
            
            end
            dw[:,:,i,j] = total
            total = zeros(size(dw)[1:2])
        end
        
        for k in 1:input_num
            bia = dout[:,:,j,k]
            bias += sum(bia)
        end
        db[:,j] .= bias
        bias = 0
    end

    dense.dw = dw
    dense.db = db
    
    # 입력값 미분
    
    total_1 = zeros(out_r, out_c)
    total_2 = zeros(out_r, out_c, out_d)

    for k in 1:input_num
    
        for j in 1:out_d
            for i in 1:input_d

                convol = conv_2d_backward(dout[:,:,i,k], dense.w[:,:,j,i])
                total_1 = total_1 + convol
            end
    
            total_2[:,:,j] = total_1
            total_1 = zeros(out_r, out_c)
        end
        
        dx[:,:,:,k] = total_2
        total_2 = zeros(out_r, out_c, out_d)
    end
        return dx
end

# Maxpooling layers

function pooling(input, row, column)
    
    
    input_r, input_c= size(input)
    pool = zeros(size(input))
    
    if input_r/row %1 != 0
        print("error")
    end

    result = []
    temp = collect(1:row:input_r)
    
    for i in temp
        for j in temp
                temp2 = (input[i:i+row-1, j:j+column-1])
                max_val = maximum(temp2)
                append!(result, max_val)
            
                if sum(temp2 .== max_val) > 1
    
                    number = []

                    for i in 1:row
    
                        num = collect(i:row:length(temp2))
                        append!(number,num)
                    end
                
                    answer = zeros(size(temp2))
    
                    for i in 1:length(temp2)
                        if temp2[number[i]] == max_val
                            answer[number[i]] = 1
                            break
                        end
                    end
    
                    pool[i:i+row-1, j:j+column-1] = answer

                else
                    pool[i:i+row-1, j:j+column-1] = temp2 .== max_val
                end
        end
    end
    
    out = reshape(result,Int(input_r/row), Int(input_c/column))'
    
    return out, pool
end

function maxpooling2D_forward_batch(grad, input, row, column)
    
    input_r, input_c, input_d, input_num = size(input)
    num_1 = Int(trunc(input_r/row))
    num_2 = Int(trunc(input_c/column))
    result = zeros(num_1,num_2,input_d,input_num)
    grad.mask = zeros(input_r, input_c,input_d,input_num)
    
    for k in 1:input_num
        
        data = input[:,:,:,k]
    
        for i in 1:input_d
            temp = pooling(data[:,:,i][1:num_1*row, 1:num_2*column], row, column)
            result[:,:,i,k] = temp[1]
        
            if size(temp[2])[1] == input_r
                grad.mask[:,:,i,k] = temp[2]
            else
                pad = zeros(input_r, input_c)
                pad[1:size(temp[2])[1],1:size(temp[2])[2]] = temp[2]
                grad.mask[:,:,i,k] = pad
            end
        end
    end
    
    return result
end

function maxpooling2D_backward_batch(grad, input ,row, column)

    input_r, input_c, input_d, input_num = size(grad.mask)
    temp = (collect(1:row:input_r))[1:size(input)[1]]
    dx = zeros(input_r, input_c, input_d, input_num)
    dd = zeros(input_r, input_c, input_d)
    
    for p in 1:input_num
        
        for k in 1:input_d
            dx_one = zeros(input_r, input_c)
        
            for i in temp
                for j in temp
                
                    dx_one[i:i+row-1,j:j+column-1] .= input[:,:,k,p][Int((i+(row-1))/row), Int((j+(column-1))/column)]
                end
            end
        
            dd[:,:,k] = dx_one
        end
    dx[:,:,:,p] = dd  
    
    end
        
    dout = dx .* grad.mask
    
    return dout
end
