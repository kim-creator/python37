class MulLayer:

  # 딥러닝 레이어 초기화(생성자에서)는 레이어 전체에서 사용할 옵션이나 변수를 미리 준비

  def __init__(self):

​

    # 여기서는 변수만 만들어 놓기

    self.x = None

    self.y = None

​

  # 곱셈 레이어에서는 역전파에 필요한 변수를 저장.

  def forward(self, x, y):

​

    # 생성자에서 만들어 놓은 변수에 값 넣기

    self.x = x

    self.y = y

​

    # 순전파 연산

    out = x * y

    return out

​

  # dout : 다음 층에서 흘러 들어오는 미분값

  def backward(self, dout):

    dx = dout * self.y

    dy = dout * self.x

​

    return dx, dy

​

# 순전파

apple = 100 # 사과 한개당 가격

apple_cnt = 2 # 사과 개수

tax = 1.1 # 소비세

​

# 계층은 2개

# (apple * apple_cnt) * tax

​

mul_apple_layer = MulLayer() # 사과 전체 가격을 구할 레이어

mul_tax_layer = MulLayer() # 소비세 까지 적용시킨 가격을 구할 레이어

​

# 순전파 수행

# 순서가 굉장히 중요합니다!!!

# 계획한 순서 그대로 레이어를 배치해서 연산을 해야 한다.

# 역전파 할 때가 문제가 된다.

​

# 순전파 때 A-B-C 순으로 계산을 했다면

# 역전파 때 C-B-A 순으로 역전파가 되어야 한다.

​

apple_price = mul_apple_layer.forward(apple, apple_cnt)

price       = mul_tax_layer.forward(apple_price, tax)

​

print("최종 사과의 가격 : {:.0f}".format(price))

​

# 역전파 수행하기

# 제일 마지막 값에 대한 미분값을 생각하기

# d돈통 / d포스기 = 1

dprice = 1

​

dapple_price, dtax = mul_tax_layer.backward(dprice)

dapple, dapple_cnt = mul_apple_layer.backward(dapple_price)

​

print("사과 전체 가격에 대한 미분값 d돈통/d사과전체가격 : {}".format(dapple_price))

print("사과 1개 가격에 대한 미분값 d돈통/d사과1개가격 : {}".format(dapple))

print("사과 개수에 대한 미분값 d돈통/d사과개수 : {}".format(dapple_cnt))

print("소비세에 대한 미분값 d돈통/d소비세 : {}".format(dtax))

​

class AddLayer:

  def __init__(self):

    # 할 거 없다.

    pass

  

  def forward(self, x, y):

    out = x + y

    return out

​

  def backward(self, dout):

    dx = dout * 1

    dy = dout * 1

​

    return dx, dy

​

apple = 100 # 사과 1개 가격

apple_cnt = 2 # 사과 개수

​

orange = 150 # 오렌지 1개 가격

orange_cnt = 3 # 오렌지 개수

​

tax = 1.1 # 

​

# 1 계층 - 사과에 대한 국소적 계산, 오렌지에 대한 국소적 계산

# (사과 1개 가격 * 사과 개수), (오렌지 1개 가격 * 오렌지 개수)

mul_apple_layer = MulLayer()

mul_orange_layer = MulLayer()

​

# 2 계층 - 사과 전체 가격 + 오렌지 전체 가격

add_apple_orange_layer = AddLayer()

​

# 3 계층 - 소비세 적용

mul_tax_layer = MulLayer()

​

# 순전파

​

# 1계층 계산

#  사과에 대한 국소적 계산

apple_price = mul_apple_layer.forward(apple, apple_cnt)

​

#  오렌지에 대한 국소적 계산

orange_price = mul_orange_layer.forward(orange, orange_cnt)

​

# 2계층 계산 ( 사과 전체 가격 + 오렌지 전체 가격 )

total_price = add_apple_orange_layer.forward(apple_price, orange_price)

​

# 3계층 계산 ( 소비세 적용 )

price = mul_tax_layer.forward(total_price, tax)

​

# 1. 전체 가격

print("전체 가격 : {}".format(price))

​

# 역전파

dprice = 1 # d돈통 / d포스기

​

# dprice/dtotal_price, dprice / dtax

dtotal_price, dtax = mul_tax_layer.backward(dprice)

​

# d돈통 / dapple_price, d돈통 / dorange_price

dapple_price, dorange_price = add_apple_orange_layer.backward(dtotal_price)

​

# 사과와 오렌지에 대한 각각의 미분값(국소적 미분)

dapple, dapple_cnt = mul_apple_layer.backward(dapple_price)

dorange, dorange_cnt = mul_orange_layer.backward(dorange_price)

​

print("사과 2개, 오렌지 3개의 가격 (소비세 적용) : {}".format(price))

print("사과 전체 가격 미분 : {}".format(dapple_price))

print("사과 개수 미분 : {}".format(dapple_cnt))

print("사과 가격 미분 : {}".format(dapple))

​

print("오렌지 전체 가격 미분 : {}".format(dorange_price))

print("오렌지 개수 미분 : {}".format(dorange_cnt))

print("오렌지 가격 미분 : {}".format(dorange))

​

print("소비세 미분 : {}".format(dtax))

​

class ReLU:

  def __init__(self):

​

    # mask : 순전파 시에 음수였던 인덱스를 저장하기 위함

    # mask가 있어야 순전파 때 음수였던 부분을 역전파 때 0으로 만들어 줄 수 있다.

    self.mask = None

​

  def forward(self, x):

    self.mask = (x <= 0) # 음수면 True, 양수면 False

    

    out = x.copy() # 원본 배열 복사

    out[self.mask] = 0 # 음수 였던 부분만 0이 된다.

​

    return out

​

  def backward(self, dout):

    # 순전파 때 음수였던 부분을 0으로 만들었다.

    # 음수였었던 인덱스를 기억하고 있다가(self.mask) 미분값 전달 시에 해당 인덱스를 0으로 만든다.

    dout[self.mask] = 0

    dx = dout

​

    return dx

​

import numpy as np

​

x = np.array([[1.0, -0.5],

              [-2.0, 3.0]])

​

relu = ReLU()

relu.forward(x)

​

relu.mask

​

dout = np.array([[-0.1, 3.0],

                 [1.3, -1.1]])

​

relu.backward(dout)

​

class Sigmoid:

​

  def __init__(self):

    # y를 out이라고 하자.

    self.out = None

  

  def forward(self, x):

    out = 1 / ( 1 + np.exp(-x))

    self.out = out

​

    return out

​

  def backward(self, dout):

    dx = dout * self.out * (1 - self.out)

    return dx
