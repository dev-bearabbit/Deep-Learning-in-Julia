# 신경망 층에 따른 가중치와 편향 설정

W = ["W1", "W2"]
b = ["b1", "b2"]
output_shape = [(1, 784), (1,50)]
weight_size = [(784, 50), (50,10)]

# 네트워크 생성하는 함수 정의

```매개변수 설명
W: 신경망에 필요한 가중치 ( 배열로 구성, ex. 2층 신경망 = [“W1”, “W2”] )
b: 신경망에 필요한 편향 ( 배열로 구성, ex. 2층 신경망 = [“b1”, “b2”] )
weight_size: 은닉층의 사이즈 ( 배열로 구성 )
output_shape: 각 층의 결과값 사이즈 ( 배열로 구성 )
weight_init: 초기값 설정 ( “std”, “Xavier”, “He” 중 선택 )
```

function making_network(W, b, weight_size, output_shape, weight_init)
    
    params = Dict()
    
    if weight_init == "std"
        for i in (1:length(W))
            params[W[i]] = 0.01 * randn(Float64, weight_size[i])
            params[b[i]] = zeros(Float64,1, weight_size[i][end])
        end
        return(params)
        
    elseif weight_init == "Xavier"
        for i in (1:length(W))
            params[W[i]] = ((1.0 / prod(output_shape[i]))^(1/2)) * randn(Float64, weight_size[i])
            params[b[i]] = zeros(Float64,1, weight_size[i][end])
        end
        return(params)
        
    elseif weight_init == "He"
        for i in (1:length(W))
            params[W[i]] = ((2.0 /prod(output_shape[i]))^(1/2)) * randn(Float64, weight_size[i])
            params[b[i]] = zeros(Float64,1, weight_size[i][end])
        end
        return(params)
    end
    return(params)
end

# 사용법

params = making_network(W, b, weight_size, output_shape, "He")
