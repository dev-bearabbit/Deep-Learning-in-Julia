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
  you can totally understand deep Learning's process.
  
- 만든 코드로 모델을 만들고 학습해본다.  
  you can make and train model using your own code.

### 코드 맛보기 (A sneak peek of the project)

간단하게 손글씨 숫자를 맞추는 분류 모델을 만들어보겠습니다.   
Let's start making model that classify handwritten digits. 

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
3.enter Julia in command line to open Julia REPL.

**4.마지막으로 아래의 코드를 입력합니다.**
4.enter the code below.

```Julia
Julia> include("train_convNet.jl")
```

