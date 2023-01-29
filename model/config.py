import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# METADATA_PATH = '/kaggle/input/edges2shoes-dataset/metadata.csv'
# IMG_PATH = '/kaggle/input/edges2shoes-dataset/'
CHECKPOINT_PATH = "model/checkpoints/"
GEN_CHECKPOINT_PATH = CHECKPOINT_PATH + "gen.pth.tar"
DISC_CHECKPOINT_PATH = CHECKPOINT_PATH + "disc.pth.tar"
IMAGES_TO_PROCESS_PATH = "IMAGES_TO_PROCESS/"
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_WORKERS = 2
LEARNING_RATE = 2e-4
BETA_1 = 0.5
BETA_2 = 0.999
NUM_EPOCHS = 16
BATCH_SIZE = 4
IMAGE_SIZE = 256
SAVE_MODEL = True


transform_edge = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.0),
        ToTensorV2()              
    ]
)

transform_both = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5)                
    ],
    additional_targets={'image0':'image'}
)
        
transform_input = A.Compose(
    [
        A.ColorJitter(p=0.1),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.0),
        ToTensorV2()
    ]
)
        
transform_target = A.Compose(
    [
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.0),
        ToTensorV2()            
    ]
)