import torch
import torch.nn as nn
import torchvision.models as models
import resnet
from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
import time

##REPRODUCIBILITY
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#args = parse_args()
#CUDA_DEVICES = args.cuda_devices
#DATASET_ROOT = args.path
CUDA_DEVICES = 0
DATASET_ROOT = './train'

init_lr = 0.01

def adjust_lr(optimizer, epoch):
	lr = init_lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def train():
	data_transform = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
    
	resnet152 = resnet.resnet152(pretrained=True)
	fc_features = resnet152.fc.in_features
	resnet152.fc = nn.Linear(fc_features,38)
    
	#print(DATASET_ROOT)
	train_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)
	data_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True, num_workers=1)
	#print(train_set.num_classes)
	
	model = resnet152
	model = model.cuda(CUDA_DEVICES)
	model.train()

	best_model_params = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	num_epochs = 15
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(params=model.parameters(), lr=init_lr, momentum=0.85)
    

	with open('TrainingAccuracy.txt','w') as fAcc:
		print('Accuracy\n', file = fAcc)
	with open('TrainingLoss.txt','w') as fLoss:
		print('Loss\n', file = fLoss)

	for epoch in range(num_epochs):
		localtime = time.asctime( time.localtime(time.time()) )
		
		print('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,num_epochs,localtime))
		print('-' * len('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,num_epochs,localtime)))

		training_loss = 0.0
		training_corrects = 0
		adjust_lr(optimizer, epoch)

		for i, (inputs, labels) in enumerate(data_loader):
			inputs = Variable(inputs.cuda(CUDA_DEVICES))
			labels = Variable(labels.cuda(CUDA_DEVICES))			

			optimizer.zero_grad()

			outputs = model(inputs)
			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			training_loss += float(loss.item() * inputs.size(0))
			#revise loss.data[0]-->loss.item()
			training_corrects += torch.sum(preds == labels.data)
			#print(f'training_corrects: {training_corrects}')

		training_loss = training_loss / len(train_set)
		training_acc =training_corrects.double() /len(train_set)
		#print(training_acc.type())
		#print(f'training_corrects: {training_corrects}\tlen(train_set):{len(train_set)}\n')
		print('Training loss: {:.4f}\taccuracy: {:.4f}\n'.format(training_loss,training_acc))

		
		if training_acc > best_acc:
			best_acc = training_acc
			best_model_params = copy.deepcopy(model.state_dict())
		
		with open('TrainingAccuracy.txt','a') as fAcc:
			print('{:.4f} '.format(training_acc), file = fAcc)
		with open('TrainingLoss.txt','a') as fLoss:
			print('{:.4f} '.format(training_loss), file = fLoss)
        
		if (epoch + 1) % 10 == 0:
			torch.save(model, 'model-epoch-{:2d}-train.pth'.format(epoch + 1))


	model.load_state_dict(best_model_params)
	torch.save(model, 'model-{:.2f}-best_train_acc.pth'.format(best_acc))


if __name__ == '__main__':
	train()
