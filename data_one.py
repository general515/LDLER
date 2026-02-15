from torchvision import datasets,transforms
import torch
from auto_augment import rand_augment_transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import pandas as pd
import random
import os
import numpy as np
import torchvision
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomDownSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.labels = np.array(dataset.label)
        self.num_classes = len(np.unique(self.labels))

        num_each_class = []
        for i in range(self.num_classes):
            num_class_i = (self.labels == i).sum()
            num_each_class.append(num_class_i)
        self.num_each_class = np.array(num_each_class).min()

        self.num_samples = self.num_each_class * self.num_classes

    def __iter__(self):
        idxs = []
        for i in range(self.num_classes):
            idxs_i = np.where(self.labels == i)[0]
            idxs_i = np.random.choice(idxs_i, self.num_each_class, replace=False)
            idxs += idxs_i.tolist()

        random.shuffle(idxs)

        return (idx for idx in idxs)

    def __len__(self):
        return self.num_samples


# RAF-DB Dataset
class RafDataset(Dataset):
    def __init__(self, raf_path, phase='train', transform=None, noisy=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1
        if noisy != None:
            assert noisy >= 0 and noisy <= 1, 'noisy is out of 0 and 1'
            to_num = len(self.label)
            noise_num = int(np.round(to_num * noisy))
            select = np.random.choice(to_num, noise_num)
            for idx in select:
                flag = True
                while flag:
                    nl = np.random.randint(0, 7)
                    if self.label[idx] != nl:
                        self.label[idx] = nl
                        flag = False
        # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path)
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx

class AffectDataset_7label(Dataset):
    def __init__(self, aff_path, phase='train', transform=None):
        self.transform = transform
        self.aff_path = aff_path
        self.imgpath = os.path.join(aff_path,'imgs')
        if phase=='train':
            self.df =pd.read_csv(os.path.join(aff_path,'info/train.csv'))
        else:
            self.df = pd.read_csv(os.path.join(aff_path, 'info/validation.csv'))
        self.imgnames = self.df.loc[:, 'img_name'].values
        self.label = self.df.loc[:, 'expression'].values
        self.imgnames = np.array(self.imgnames)
        self.label = np.array(self.label)
        idxs = np.where(self.label != 7)[0]
        self.imgnames = self.imgnames[idxs].tolist()
        self.label = self.label[idxs].tolist()
        assert len(self.imgnames) == len(self.label), 'Number of images not equals that of labels.'

    def __len__(self):
        return len(self.imgnames)

    def __getitem__(self, idx):
        path = os.path.join(self.imgpath, self.imgnames[idx])
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx

class FERPlus(torchvision.datasets.ImageFolder):
    def __init__(self, data_path=r'D:\datasets\fer2013_plus_combine', phase='train', transform=None):
        if phase == 'train':
            super().__init__(os.path.join(data_path, 'Training'), transform=transform)
        else:
            super().__init__(os.path.join(data_path, 'PrivateTest'), transform=transform)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx

def get_dataloaders(dataset='raf', data_path='./datasets/raf-basic',
                    batch_size=64, num_workers=8):
    # transforms
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            rand_augment_transform(config_str='rand-m5-n3-mstd0.5', hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(scale=(0.02, 0.25)),
            ])
    data_transforms_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])
    if dataset == 'raf':
        dataset = RafDataset
    elif dataset == 'ferplus':
        dataset = FERPlus
    elif dataset == 'affectnet':
        dataset = AffectDataset_7label

    train_dataset = dataset(
            data_path,
            phase='train',
            transform=data_transforms
        )
    val_dataset = dataset(
            data_path,
            phase='test',
            transform=data_transforms_val
    )

    train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                drop_last=False, #drop_last must be False
                persistent_workers=True,
            )
    val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            persistent_workers=True
        )

    return train_loader, val_loader

if __name__=='__main__':
    tl, vl = get_dataloaders(dataset='ferplus',data_path=r'D:\datasets\fer2013_plus')
    #tl, vl = get_dataloaders(dataset='affectnet', data_path=r'E:\datasets\AffectNet\processed')
    #tl, vl = get_dataloaders(dataset='raf',data_path=r'E:/BaiduWangpan/RAF/basic')
    print('a')