from dataset import CUB200
from torch.utils.data.dataloader import DataLoader

dataset = CUB200(".", download=True)
loader = DataLoader(dataset, 16, 4)
