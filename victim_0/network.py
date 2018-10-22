from torch import nn
from torch.autograd import Variable
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import utils as vutils, datasets, transforms
from torch import nn

IMG_SIZE=32

# Utils
FILE_TYPE = ".checkpoint"
def save_checkpoint(filepath, state, best=None):
    torch.save(state, filepath)
    if best is not None:
        shutil.copyfile(filepath,
                        os.path.join(
                            os.path.dirname(filepath),
                            'best.pth.tar'
                            )
                        )


def load_checkpoint(filepath, args=None):
    if args is None:
        return torch.load(filepath)
    elif args.cuda:
        return torch.load(filepath)
    else:
        return torch.load(filepath, map_location='cpu')


def get_victim(filepath):
    checkpoint = load_checkpoint(filepath)
    model = TrafficSignClassifier()
    model.load_state_dict(checkpoint['model'])
    return model.cuda()

# Network
class Classifier(nn.Module):
    def __init__(self, input_nbr, out_nbr):
        super(Classifier, self).__init__()
        self.input_nbr = input_nbr
        self.lin = nn.Linear(input_nbr, out_nbr)

    def forward(self, x):
        return self.lin(x)

class SoftMaxClassifier(Classifier):
    def __init__(self, in_len, out_len):
        super().__init__(in_len, out_len)

    def forward(self, x):
        x = super().forward(x)
        return nn.functional.log_softmax(x)

class TrafficSignClassifier(nn.Module):
    def __init__(self):
        super(TrafficSignClassifier, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 2, stride=2, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(0.2),
            # 16*16*16
            torch.nn.Conv2d(16, 32, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(0.2),
            # 32*8*8
            torch.nn.Conv2d(32, 64, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            # 64*4*4
            torch.nn.Conv2d(64, 128, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            # 128*2*2
            )
        self.final = torch.nn.Sequential(
            torch.nn.Linear(128*2*2, 128*2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(128*2, 43,bias=False),
            # 1*1*1
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.final(x)
        return torch.nn.functional.log_softmax(x,dim=0)


# Training 
class Trainer():
    def __init__(self,train_loader,val_loader):
        self.model = TrafficSignClassifier()
        self.model = self.model.cuda()
        self.train_loader=train_loader
        self.val_loader=val_loader
    def load(self,filepath):
        self.model = get_victim(filepath)
        self.model = self.model.cuda()
    def train(self):
        self.epochs = 10000
        criterion = nn.CrossEntropyLoss()
        start_epoch = 0
        self.cuda_available=True
        self.model.train()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,self.model.parameters()),lr=0.001)
        print("Starting training")
        for epoch in range(start_epoch, self.epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images_batch = Variable(images)
                labels_batch = Variable(labels)

                if self.cuda_available:
                    images_batch = images_batch.cuda()
                    labels_batch = labels_batch.cuda(async=True)

                self.optimizer.zero_grad()
                output = self.model(images_batch)
                loss = criterion(output, labels_batch.long())
                loss.backward()
                self.optimizer.step()

                if i%500==0:
                    print(('Epoch: [{0}], Step: [{1}/{2}], Loss: {3},')
                          .format(epoch + 1,
                                  i + 1,
                                  len(self.train_loader),
                                  loss.item()))

            # train_acc, train_loss = self.validate_model(self.train_loader, self.model)
            val_acc, val_loss = self.validate_model(self.val_loader, self.model)
            #print(train_acc,train_loss)
            print(val_acc, val_loss)

            if(val_acc>95):
                checkpoint = {}
                checkpoint['model'] = self.model.state_dict()
                save_checkpoint('working_model_{}.chk'.format(val_acc),checkpoint)
                save_checkpoint('working_model_latest.chk'.format(val_acc),checkpoint)

    def validate_model(self, loader, model):
        model.eval()
        correct = 0.0
        total = 0.0
        total_loss = 0.0
        i = 0
        for images, labels in loader:
            i+=1
            images_batch = torch.tensor(images)
            labels_batch = Variable(labels)

            if self.cuda_available:
                images_batch = images_batch.cuda()
                labels_batch = labels_batch.cuda()

            output = model(images_batch)
            loss = nn.functional.cross_entropy(output, labels_batch.long(), size_average=False)
            total_loss += loss.item()
            total += len(labels_batch)

            if not self.cuda_available:
                correct += (labels_batch == output.max(1)[1]).data.cpu().numpy().sum()
            else:
                correct += (labels_batch==output.max(1)[1]).detach().cpu().numpy().sum()
        model.train()

        average_loss = total_loss / total
        return correct / (total+0.0) * 100.0, average_loss



def main():
    data_transforms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),transforms.ToTensor(),])
    dataset = datasets.ImageFolder('../training/GTSRB/Final_Training/val_images', transform=data_transforms)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    validation_split = 0.2
    random_seed= 42
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                    sampler=valid_sampler)

    trainer = Trainer(train_loader,validation_loader)
    trainer.load('working_model_latest.chk')
    trainer.train()

if __name__ == '__main__':
    main()
