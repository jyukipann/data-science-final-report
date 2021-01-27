import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score ##############

from my_utility import models
import time
import os

def classification_train():
	# cudaが利用できるなら一つ目のGPUが，そうでなければcpuを使うようにする
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	img_transform = transforms.Compose([
		transforms.ToTensor()
	])

	traindataset = datasets.FashionMNIST('./data', transform=img_transform, train=True,download=False)  # 一度端末に保存したらdownloadはFalseにしておきましょう
	trainloader = DataLoader(traindataset, batch_size=100, shuffle=True)

	net = models.reportCNN(10)
	net.to(device)

	optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
	criterion = torch.nn.NLLLoss()

	print(net)

	num_epoch = 5
	since = time.time()

	for epoch in range(num_epoch):
		running_loss = 0.0
		running_acc = 0.0
		total = 0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			##ここで精度計算
			results = outputs.cpu().detach().numpy().argmax(axis=1)
			running_acc += accuracy_score(labels.cpu().numpy(), results) * len(inputs)
			total += len(inputs)

			running_loss += loss.item()

		running_acc /= total
		print('Loss: ',running_loss, 'ACC :', running_acc, epoch)

	print('Finished Training')

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

	if not os.path.exists('./save_models'):
		os.makedirs('./save_models')

	torch.save(net.state_dict(), './save_models/fashion_mnist_classification.pth')