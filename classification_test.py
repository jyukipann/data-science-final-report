import torch
import torch.nn.functional as F

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from my_utility import models

def classification_test():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	img_transform = transforms.Compose([
		transforms.ToTensor()
	])

	testdataset = datasets.FashionMNIST('./data', transform=img_transform, train=False,download= False)  # 一度端末に保存したらdownloadはFalseにしておきましょう
	testloader = DataLoader(testdataset, batch_size=100, shuffle=False)

	net = models.reportCNN(10)
	net.load_state_dict(torch.load('./save_models/fashion_mnist_classification.pth'))
	net.to(device)
	net.eval()

	test_loss = 0
	target_all=[]
	result_all=[]

	with torch.no_grad():
		for data, target in testloader:
			data, target = data.to(device), target.to(device)
			output = net(data)
			pred_cpu = output.cpu().detach().numpy().argmax(axis=1)
			test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
			# すべてのテストデータの正解ラベルと推定されたラベルをnumpy配列に格納
			target_all = np.append(target_all, target.cpu().numpy())
			result_all = np.append(result_all, pred_cpu)

	test_loss /= len(testloader.dataset)
	acc = accuracy_score(target_all, result_all)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, int(acc*len(testloader.dataset)), len(testloader.dataset),
		100. * acc))

	c_mat= confusion_matrix(target_all, result_all)
	print(c_mat)