from dataset import CUB200
from torch.utils.data.dataloader import DataLoader
from torch.utils.trainer import Trainer

from torch import nn, optim
from torch.autograd import Variable
from torchvision.models import resnet18
import torch.nn.functional as F

NUM_EPOCHS = 1
BATCH_SIZE=2

dataset = CUB200(".", download=False, nb_examples=100)
loader = DataLoader(dataset, BATCH_SIZE)

model = resnet18()
model.fc = nn.Linear(512,200)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

trainer = Trainer(model, criterion, optimizer, loader)
trainer.run(NUM_EPOCHS)
