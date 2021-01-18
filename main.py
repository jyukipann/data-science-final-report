from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import random

#class
N = 4

#task1
X, y = make_blobs(random_state=122, n_samples=450, n_features=2, cluster_std=1.8, centers=N)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(type(X),X.shape)
print(type(y),y.shape)


colors = ["deeppink", "lawngreen", "blue", "salmon"]
if N > 4:
	from matplotlib import colors as mcolors
	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()
	colors = random.sample(colors, N)

plt.grid(True)
plt.title("This is a title")
plt.xlabel("x axis")
plt.ylabel("y axis")
for i in range(N):
	plt.scatter(X[y == i][:, 0],X[y == i][:, 1], c=colors[i], alpha=0.3)
plt.show()

"""
svm = SVC(C=4, gamma="auto")
svm.fit(X_train, y_train)
"""
estimator = MLPClassifier()
estimator.fit(X_train, y_train)


"""
#task2
X_org, y_org = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False) 
"""


"""
１．クラス分類
以下の条件でデータを生成し，これらを授業で教えた分類器で分類し，テストデータに対する正解率
（accuracy_score）および，テストデータと分類境界線がわかる図を示せ．またテストデータに対する混
同行列を示せ．
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
X, y = make_blobs(random_state=122, n_samples=450, n_features=2, cluster_std=1.8, centers=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

２．次元削減
0 から９の手書き数字画像データセットである MNIST データから適当に 3000 個のデータを選択し，
各数字の頻度を示せ．選択したデータを T-SNE により２次元に次元削減をおこなえ．結果をプロットす
る際には，同じクラスのデータが同じ色で表示されるようにせよ．１つのクラスタとしてまとまってい
ない数字を示し，その理由を考察せよ．
MNIST データのダウンロードは以下の様にできる．
from sklearn.datasets import fetch_openml
X_org, y_org = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False) 

３．深層学習
Fashion-MNIST に対して，深層学習による分類をおこなえ．使用したネットワークの構成について説
明せよ．学習データに対する損失関数の値と正解率の推移，テストデータに対する正解率と混同行列を
しめし，結果を考察せよ．なお pytorch では，第１４回に配布したプログラムにおいて，datasets.MNIST
を datasets.FashionMNIST にするだけで利用可能になる．
"""