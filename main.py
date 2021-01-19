from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib.colors import ListedColormap




def gene_colors(N=5):
	from matplotlib import colors as mcolors
	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()
	colors = random.sample(colors, N)
	return colors

#task1
"""
１．クラス分類
以下の条件でデータを生成し，これらを授業で教えた分類器で分類し，テストデータに対する正解率
（accuracy_score）および，テストデータと分類境界線がわかる図を示せ．またテストデータに対する混
同行列を示せ．
"""
def task1():
	#class
	N = 4

	X, y = make_blobs(random_state=122, n_samples=450, n_features=2, cluster_std=1.8, centers=N)
	X_train, _, y_train, _ = train_test_split(X, y, random_state=1)

	colors = ["deeppink", "lawngreen", "blue", "salmon"]
	if N > 4:
		colors = gene_colors(N=N)

	plt.grid(True)
	plt.title("This is a title")
	plt.xlabel("x axis")
	plt.ylabel("y axis")
	for i in range(N):
		plt.scatter(X[y == i][:, 0],X[y == i][:, 1], c=colors[i], alpha=0.8)


	if True:
		estimator = SVC(C=4, gamma="auto")
	else:
		estimator = MLPClassifier()

	estimator.fit(X_train, y_train)
	print(estimator.score(X_train,y_train))

	x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
	y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
	resolution = 0.5
	x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, resolution),np.arange(y_min, y_max, resolution))
	z = estimator.predict(np.array([x_mesh.ravel(), y_mesh.ravel()]).T)
	z = z.reshape(x_mesh.shape)
	cmap = ListedColormap(tuple(colors))
	plt.contourf(x_mesh, y_mesh, z, alpha=0.4, cmap=cmap)
	plt.xlim(x_mesh.min(), x_mesh.max())
	plt.ylim(y_mesh.min(), y_mesh.max())
	plt.show()


#task2
"""２．次元削減
0 から９の手書き数字画像データセットである MNIST データから適当に 3000 個のデータを選択し，
各数字の頻度を示せ．選択したデータを T-SNE により２次元に次元削減をおこなえ．結果をプロットす
る際には，同じクラスのデータが同じ色で表示されるようにせよ．１つのクラスタとしてまとまってい
ない数字を示し，その理由を考察せよ．
"""
def task2():
	from sklearn.datasets import fetch_openml
	X_org, y_org = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True, as_frame=False)
	X = X_org[0:3000]
	y = y_org[0:3000]
	import collections
	c = collections.Counter(y)
	print(c)

	from sklearn.manifold import TSNE
	X = TSNE(n_components=2).fit_transform(X)

	colors = gene_colors(N=10)
	plt.clf()
	for i in range(10):
		plt.scatter(X[y == str(i)][:, 0],X[y == str(i)][:, 1], c=colors[i], alpha=0.8, label=str(i))
	plt.legend(loc='upper left')
	plt.show()

def task3():
	pass


if __name__ == "__main__":
	task = [task1,task2,task3]
	t = None
	while True:
		try:
			t = task[int(input("select task 1-3 : "))-1]
		except:
			exit(0)
		t()


"""

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
X, y = make_blobs(random_state=122, n_samples=450, n_features=2, cluster_std=1.8, centers=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


from sklearn.datasets import fetch_openml
X_org, y_org = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False) 

３．深層学習
Fashion-MNIST に対して，深層学習による分類をおこなえ．使用したネットワークの構成について説
明せよ．学習データに対する損失関数の値と正解率の推移，テストデータに対する正解率と混同行列を
しめし，結果を考察せよ．なお pytorch では，第１４回に配布したプログラムにおいて，datasets.MNIST
を datasets.FashionMNIST にするだけで利用可能になる．
"""