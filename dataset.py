from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import os
import pandas as pd

class ImagenetValidationDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, idx_by_name=False):
        self.image_labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.idx_by_name = idx_by_name

    def __len__(self):
        # return len(self.image_labels)
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        # if indices passed as tensors - transform to list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # generates proper name for the images
        if self.idx_by_name:
            img_idx = str(idx+1).zfill(8)
            img_name = os.path.join(self.root_dir,
                                    "ILSVRC2012_val_"+img_idx+".JPEG")
        else:
            img_list = sorted(os.listdir(self.root_dir))
            img_name = os.path.join(self.root_dir, img_list[idx])
        
        # open the image and transform
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # get the label
        label = self.image_labels.iloc[idx, 0]
        
        # return image and label
        sample = {'image': image, 'label': label}
        return sample

def get_dataset(csv_file='ImageNet/validation_labels.csv',
                root_dir='ImageNet/val_images/'):
    # standard ImageNet transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # get instance of the dataset
    dataset = ImagenetValidationDataset(csv_file=csv_file,
                                        root_dir=root_dir,
                                        transform=preprocess)
    
    return dataset