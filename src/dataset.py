from torchvision import datasets,transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pathlib import Path
from torch.utils.data import DataLoader
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
train_path = PROJECT_ROOT / "data" / "train"
ood_path = PROJECT_ROOT / "data" / "ood_validation"


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
])


train_data = datasets.ImageFolder(train_path,transform=transform)

ood_data = datasets.ImageFolder(ood_path,transform=transform)

train_indices, val_indices = train_test_split(
    range(len(train_data)),
    test_size=0.2,
    random_state=SEED,
    stratify=train_data.targets
)

train_dataset = Subset(train_data , train_indices)
val_dataset = Subset(train_data , val_indices)

ood_dataset = ood_data

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


images,labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)
print(labels[:10])