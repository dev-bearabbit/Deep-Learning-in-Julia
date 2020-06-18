using MLDatasets
train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata();
train_x = Array(reshape(train_x, 28,28,1,60000));
test_x = Array(reshape(test_x,28,28,1,10000));

function making_one_hot_label(x, y)
"""
    원-핫 인코딩 레이블을 만드는 함수
    예를 들어 0~9까지의 숫자 중 3을 원-핫 레이블로 만들면
    [0  0  0  1  0  0  0  0  0  0]과 같이 출력할 것이다.
    x : 만들려는 숫자
    y: 메트릭스의 길이, 주의할 점은 이것은 0부터 시작한다!
"""
    temp = x + 1
    temp_matrix = zeros(Int8, 1, y)
    temp_matrix[temp] = 1
    return(temp_matrix)
end

function making_one_hot_labels(y_train)
    t = making_one_hot_label.(y_train, 10)
    return (reduce(vcat, t))
end

# t 자료를 원핫 라벨로 변경합니다.
t = making_one_hot_labels(train_y)
typeof(t), size(t)
