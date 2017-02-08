import numpy as np
from PIL import Image # pillow

# 활성화함수
# 활성화 함수로는 비선형을 사용해야. 선형을 사용하면 층을 쌓는 의미가 없어짐
def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)

#출력 총합이 1 -> 확률로 해석
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # for overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 그래프 그리기
# import matplotlib.pyplot as plt
# from matplotlib.image import imread
#
# x = np.arange(0, 6, 0.1)
# y = np.sin(x)
#
# plt.plot(x, y)
# plt.show()
#
# img = imread('./lena.png')
#
# plt.imshow(img)
# plt.show()

# x = np.arange(-5, 5, 0.1)
# y = sigmoid(x)
#
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# # plt.show()
#
# x = np.array([1, 2])
#
# print(np.ndim(x))
# print(x.shape)
# print(x.shape[0])
#