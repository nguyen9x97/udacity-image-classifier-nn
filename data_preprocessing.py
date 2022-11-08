from torchvision import datasets, transforms
import torch
from typing import List


def get_train_val_test_dirs(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    return train_dir, valid_dir, test_dir


def gen_data_transforms():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    return data_transforms


def load_image_datasets(train_val_test_dirs: List[str], data_transforms):
    """
    PARAMS:
        + train_val_test_dirs: [train_dir, val_dir, test_dir]
    """
    image_datasets = dict()
    for x, x_dir in zip(['train', 'val', 'test'], train_val_test_dirs):
        image_datasets[x] = datasets.ImageFolder(x_dir, transform=data_transforms[x])
        
    return image_datasets


def gen_dataloaders(image_datasets):
    dataloaders = dict()
    for x in ['train', 'val', 'test']:
        dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
        
    return dataloaders
