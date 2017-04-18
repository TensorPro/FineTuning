from dataset import CUBS200
from torch.utils.data.dataloader import DataLoader

dataset = CUBS200(".", download=True)
loader = DataLoader(dataset, 16, 4)
