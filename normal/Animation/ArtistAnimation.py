import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

ims = []

# a number fo frame is 10.
for i in range(10):
        rand = np.random.randn(100)     # 100個の乱数を生成
        print("rand = {0}".format(rand))
        im = plt.plot(rand)             # 乱数をグラフにする
        ims.append(im)                  # グラフを配列 ims に追加

# 10枚のプロットを 100ms ごとに表示
ani = animation.ArtistAnimation(fig, ims, interval=100)
plt.show()
# ani.save("output.gif", writer="imagemagick")
