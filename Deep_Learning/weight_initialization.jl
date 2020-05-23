# 신경망 층에 따른 가중치와 편향 설정

W = ["W1", "W2", "W3"]
b = ["b1", "b2", "b3"]
hidden_size = [50, 100]

# 네트워크 생성하는 함수 정의

```매개변수 설명
W와 b: 가중치와 편향 이름 배열
input_size: 데이터 개수
hidden_size: 은닉층 개수
output_size: 결과로 나와야 하는 개수
weight_init: 가중치 초기값 설정, 3개("std","Xavier","He" 중 하나 선택 가능)
```

function making_network(W, b, input_size, hidden_size, output_size, weight_init)
    
    params = Dict()
    
    if weight_init == "std"
        for i in (1:length(W))
            if i == 1
                params[W[i]] = 0.01 * randn(Float64, input_size, hidden_size[i])
                params[b[i]] = zeros(Float64, 1, hidden_size[i])
            elseif i == length(W)
                params[W[i]] = 0.01 * randn(Float64, hidden_size[i-1], output_size)
                params[b[i]] = zeros(Float64, 1, output_size)
            else
                params[W[i]] = 0.01 * randn(Float64, hidden_size[i-1], hidden_size[i])
                params[b[i]] = zeros(Float64, 1, hidden_size[i])
            end
        end
        return(params)
        
    elseif weight_init == "Xavier"
        for i in (1:length(W))
            if i == 1
                params[W[i]] = ((1.0 / input_size)^(1/2)) * randn(Float64, input_size, hidden_size[i])
                params[b[i]] = zeros(Float64, 1, hidden_size[i])
            elseif i == length(W)
                params[W[i]] = ((1.0 / hidden_size[i-1])^(1/2)) * randn(Float64, hidden_size[i-1], output_size)
                params[b[i]] = zeros(Float64, 1, output_size)
            else
                params[W[i]] = ((1.0 / hidden_size[i-1])^(1/2)) * randn(Float64, hidden_size[i-1], hidden_size[i])
                params[b[i]] = zeros(Float64, 1, hidden_size[i])
            end
        end
        return(params)
        
    elseif weight_init == "He"
        for i in (1:length(W))
            if i == 1
                params[W[i]] = ((2.0 / input_size)^(1/2)) * randn(Float64, input_size, hidden_size[i])
                params[b[i]] = zeros(Float64, 1, hidden_size[i])
            elseif i == length(W)
                params[W[i]] = ((2.0 / hidden_size[i-1])^(1/2)) * randn(Float64, hidden_size[i-1], output_size)
                params[b[i]] = zeros(Float64, 1, output_size)
            else
                params[W[i]] = ((2.0 / hidden_size[i-1])^(1/2)) * randn(Float64, hidden_size[i-1], hidden_size[i])
                params[b[i]] = zeros(Float64, 1, hidden_size[i])
            end
        end
        return(params)
    end
    return(params)
end

# 사용법

params = making_network(W, b, 784, hidden_size, 10, "He")

# 일반화된 순전파 미분 함수 코드

function numerical_gradient_forward(f, x, t)
    for i in (1:length(W))
        grads[W[i]]= numerical_gradient(f, x, t,params[W[i]])
        grads[b[i]] = numerical_gradient(f, x, t,params[b[i]])
    end
    return(grads)
end
