using MLDatasets
# train_x: 원데이터(3차원 x 50000개, 4차원), train_y_coarse: 수퍼클라스 20개 (0~19), train_y_fine: 하위 클라스 100개 (0~99)
train_x, train_y_coarse, train_y_fine = CIFAR100.traindata()
test_x, test_y_coarse, test_y_fine  = CIFAR100.testdata()

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
    t = making_one_hot_label.(y_train, 100)
    return (reduce(vcat, t))
end

t = making_one_hot_labels(train_y_fine);
