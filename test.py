import torch
from utils import parse_args
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from dataset import IMAGE_Dataset
from sklearn.metrics import f1_score
import numpy as np

CUDA_DEVICES = 0
DATASET_ROOT = './test'
PATH_TO_WEIGHTS = './model-epoch-12-train.pth'


def test():
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    test_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)
    data_loader = DataLoader(
        dataset=test_set, batch_size=32, shuffle=True, num_workers=1)
    classes = [_dir.name for _dir in Path(DATASET_ROOT).glob('*')]
    classes.sort()
    classes.sort(key = len)

    model = torch.load(PATH_TO_WEIGHTS)
    model = model.cuda(CUDA_DEVICES)
    model.eval()

    total_correct = 0
    total = 0
    f1_score = 0.0
    class_correct = list(0. for i in enumerate(classes))
    class_total = list(0. for i in enumerate(classes))
    
    # log TP TN FP FN
    
    class_F1_status = list(0. for i in enumerate(classes))
    class_F1_score = list(0. for i in enumerate(classes))
    class_TP = list(0. for i in enumerate(classes))
    class_TN = list(0. for i in enumerate(classes))
    class_FP = list(0. for i in enumerate(classes))
    class_FN = list(0. for i in enumerate(classes))
    
    classes_num = 38
    label_num = torch.zeros((1,classes_num))
    predict_num = torch.zeros((1,classes_num))
    acc_num = torch.zeros((1,classes_num))
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # totoal
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            
            # batch size
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            
            pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.) 
            predict_num += pre_mask.sum(0) 
            tar_mask = torch.zeros(outputs.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.) 
            label_num += tar_mask.sum(0) 
            acc_mask = pre_mask*tar_mask 
            acc_num += acc_mask.sum(0)
#            print('------------------')
#            print(predict_num)
#            print(label_num)
#            print(acc_num)
            
        recall = acc_num/label_num 
        precision = acc_num/predict_num 
        F1 = 2*recall*precision/(recall+precision) 
        accuracy = acc_num.sum(1)/label_num.sum(1)
        
        print(F1)
        print(accuracy)                           


    F1_num = F1.numpy()
    for i, c in enumerate(classes):
        print('Accuracy of %5s : %8.4f %%' % (
        c, 100 * class_correct[i] / class_total[i]), end = '')
        
        if np.isnan(F1_num[0][i]):
            F1_num[0][i] = 0
        print(' , f1-score of %5s : %8.4f %%' % (
        c, 100 * F1_num[0][i]))
    
    # Accuracy
    print('\nAccuracy on the ALL test images: %.4f %%'
      % (100 * total_correct / total))
    
    # f1-score
    f1_score = 100 * (F1.sum()) / classes_num
    print('Total f1-score : %4f %%' % (f1_score) )
    


if __name__ == '__main__':
    test()



