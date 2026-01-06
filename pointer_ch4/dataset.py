'''
Let's think about how I'd do this in a Colab environment. 
Let's say I load up the function: prepare_datasets. 

I will run it: prepare_datasets()

This should do the following:
- install gdown if not installed 
- download the data with gdown
- Unzip it into Data. 

Then, the necessary models will feed them into DataLoaders. 
'''
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import gdown
from pathlib import Path
import zipfile

def prepare_datasets():
    url = "https://drive.google.com/uc?id=1Z_5ncaT9yoYXEa-bKEc0GwEgZmrTK2G7"
    ROOT = Path(__file__).resolve().parent
    print("Path(__file__).resolve(): ", Path(__file__).resolve())
    print("Path(__file__).resolve().parent: ", Path(__file__).resolve().parent)
    PROJECT_DIR = ROOT / "pointer_ch4"
    PROJECT_DIR.mkdir(exist_ok=True)
    DATA_DIR = PROJECT_DIR / "data"
    output = PROJECT_DIR / "data.zip"

    gdown.download(url, str(output), quiet=False)

    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    train_data_path = './data/fish_cat_images/train'

    transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transforms)

    train_data[0][0].shape

    val_data_path = "./data/fish_cat_images/val/"
    val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=transforms)

    test_data_path = "./data/fish_cat_images/test/"
    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transforms)

    batch_size=64
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    val_data_loader = DataLoader(val_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return {
        'train_data_loader': train_data_loader,
        'val_data_loader': val_data_loader,
        'test_data_loader': test_data_loader
    }



