# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from DeepLearning.dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)


img = x_train[0]
label = t_train[0]
print('label = {0}'.format(label))  # 5

print('t_train[0].shape = {0}'.format(img.shape))  # (784,)
img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
print('t_train[0].reshape(28, 28) = {0}'.format(img.shape))  # (28, 28)

img_show(img)

print('t_tain[20].label = {0}'.format(t_train[20]))
img = x_train[20]
img = img.reshape(28, 28)
img_show(img)
