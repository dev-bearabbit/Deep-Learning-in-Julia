## 줄리아를 통해 딥러닝 100% 이해하기 프로젝트<br/>(A project for 100% understanding Deep Learning in Julia)

해당 프로젝트는 오로지 줄리아(Julia)만을 사용하여 딥러닝 모델을 구현합니다.  
This project uses only Julia language to implement deep learning model.

교육용으로 제작된 것이기에 딥러닝 원리에 대해 공부하실 분들에게 추천합니다.  
It's designed for education, so I recommend for those who want to study the principles of deep learning.

이 프로젝트에서 사용된 코드들은 아래의 저서를 참고하였습니다.  
I refered to the book below.

- [밑바닥부터 시작하는 딥러닝](https://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)

제 블로그의 [Deep Learning in Julia](https://hyeonji-ryu.github.io/categories/Deep-learning-in-Julia/) 카테고리에 구현 과정이 업로드되어 있습니다.  
I uploaded some posts about the project's process on my blog, but they are written only Korean.

P.S. 줄리아의 사용법이 궁금하다면, 제 블로그에 번역해둔 [Think Julia](https://hyeonji-ryu.github.io/categories/Think-Julia/)을 통해 공부하실 수 있습니다.

### 프로젝트 목표 (The project's goal)

- 줄리아의 사용법을 배운다.  
  you can learn how to use Julia language
  
- 딥러닝의 작동방식을 완전히 이해한다.  
  you can totally understand deep learning's process.
  
- 만든 코드로 모델을 만들고 학습해본다.  
  you can make and train model using your own code.

### 코드 맛보기 (A sneak peek of the project)

간단하게 손글씨 숫자를 맞추는 분류 모델을 만들어보겠습니다.   
Let's start making model that classify handwritten digits. 

>**NOTE**  
>우리가 오늘 사용할 모델에 대한 정보는 아래와 같습니다.  
>Here is the information about model structure and train parameters.
>
>**- model structure**  
>Conv(filter_num = 32, filter_size = 3, stride = 1, padding = 0) + ReLU  
>Max_pool(filter size = 2, stride = 1, padding = 0)  
>Conv(filter_num = 64, filter_size = 3, stride = 1, padding = 0) + ReLU  
>Max_pool(filter size = 2, stride = 1, padding = 0)  
>Conv(filter_num = 64, filter_size = 3, stride = 1, padding = 0) + ReLU  
>Max_pool(filter size = 2, stride = 1, padding = 0)  
>Flatten  
>Dense(node_num = 64) + Relu  
>Dense(node_num = 10) + softmax 
>
>**- train parameters**  
>weight initializer = He   
>optimizer = Adam  
>batch size = 100  
>epochs = 1 

#### 모델 훈련하기 (training the model)

**1. 프로젝트의 전체 코드를 다운받습니다.**  
1.download the entire code in this project. 

프로젝트 상단에서 직접 다운받을 수도 있고, git을 사용하신다면 아래의 코드를 입력해서 다운받을 수도 있습니다.  
you can download it derectly from top of this page, or you can clone it using git. 

```bash
$ git clone https://github.com/Hyeonji-Ryu/Deep_Learning_in_Julia.git
```

**2. 커맨드라인에서 CNN 폴더로 디렉토리 경로를 설정합니다.**  
2.set a directory path to CNN folder in command line.

```bash
$ cd <your path>/Deep_Learning_in_Julia/CNN
```

**3. 줄리아 REPL를 열기 위해 해당 커맨드라인에서 `Julia`를 입력합니다.**  
3. enter `Julia` in command line to open Julia REPL.

**4.마지막으로 아래의 코드를 입력합니다.**  
4.enter the code below.

```Julia
Julia> include("train_convNet.jl")
``` 

**5. 아래와 같이 로스값을 바로 확인할 수 있습니다.**  
5.you can see loss values in real time.

```Julia
NO.1: 2.2886466121074425
NO.2: 2.280258799992027
NO.3: 2.2746683599835764
NO.4: 2.2608282595650575
NO.5: 2.243700243712272
NO.6: 2.214588691106391
.
.
NO.595: 0.10943427897500864
NO.596: 0.12955831960259565
NO.597: 0.03989749763491561
NO.598: 0.013439008366053344
NO.599: 0.1109573868949143
NO.600: 0.08915640203576712
```
**6. 훈련이 끝난 후, 자동으로 테스트한 결과를 보여줍니다.**  
6.you can see accuracy of test set automatically.

```Julia
98.08
```

#### 훈련된 모델로 숫자 예측해보기 (predicting digit using trained model)

모델이 제대로 훈련되었는지 확인해봅시다.
Let's check that the model is trained well.

먼저, 훈련데이터 중 하나를 테스트 데이터로 지정합니다.
First, assign one of train set to `test`.

```julia
Julia> test = reshape(train_x[:,:,:,1],28,28,1,1)
```
다음으로 아래
