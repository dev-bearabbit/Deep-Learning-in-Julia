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
    temp = drop_out_single(input_size, rate)
    #print(temp)
    temp_num = hidden_size - 1
    for i = 1:temp_num
        temp_1 = drop_out_single(input_size, rate)
        #print(temp_1)
        temp = [temp; temp_1]
        #print(temp)
    end
    return temp
end

function dropout_forward(dropout, x, dropout_ratio)
    
    if dropout_ratio < 1
        dropout.mask = drop_out(size(x)[2],size(x)[1], dropout_ratio)
        return x .* dropout.mask
        
    else dropout_ratio = 1
        return x .*  (1.0 - dropout_ratio)
    end
end

function dropout_backward(dropout, dout)
    return dout .* dropout.mask
end
