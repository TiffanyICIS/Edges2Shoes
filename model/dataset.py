from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from model.config import transform_both, transform_input, transform_target

class ShoesDataset(Dataset):
    def __init__(self, root_dir: str, mode: str):
        self.mode = mode
        self.root_dir = root_dir + f"{mode}/"
        self.file_list = os.listdir(self.root_dir)
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        img_file = self.file_list[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = Image.open(img_path)
        
        img_w, img_h = image.size
        
        input_image_PIL = image.crop((0, 0, img_w//2, img_h))
        target_image_PIL = image.crop((img_w//2, 0, img_w, img_h))
        
        input_image = np.array(input_image_PIL)
        target_image = np.array(target_image_PIL)
        
        augmentations = transform_both(image = input_image, image0 = target_image)
        input_image, target_image = augmentations['image'], augmentations['image0']
        
        input_image = transform_input(image = input_image)['image']
        target_image = transform_target(image = target_image)['image']
        
        return input_image, target_image


# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True, num_workers=NUM_WORKERS, drop_last = True)
# val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle = False, drop_last = True)