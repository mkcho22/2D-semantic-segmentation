import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')

import torch
import torch.optim as optim
from train import train_model, EarlyStopper, evaluate_model
from data import setup_data_loaders
from utils import UNet

# 하이퍼파라미터
DATA_DIR = "./2D Segmentation"
NUM_CLASSES = 13
INPUT_HEIGHT = 600
INPUT_WIDTH = 800
patience = 10
EPOCHS = 100

BATCH_SIZE = 4
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.01
LAMBDA = 0.5 # Loss 가중치.
GAMMA = 1.0 #FocalLoss 하이퍼파라미터
ALPHA = 0.25

# SGD + Momentum 파라미터
#MOMENTUM_RATE = 0.9

# 장치 설정
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using device: CUDA (NVIDIA GPU)")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}") 
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps") 
    print(f"Using device: MPS (Apple Silicon GPU)")
else:
    DEVICE = torch.device("cpu")
    print(f"Using device: CPU")

if __name__ == '__main__':
    # 데이터 로더
    train_loader, val_loader = setup_data_loaders(BATCH_SIZE, DATA_DIR, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES)
    
    # model & optimizer
    model = UNet(n_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)

    # 학습 시작
    print("--- Starting training ---")
    train_model(model, train_loader, val_loader, optimizer, num_epochs=EPOCHS,patience=patience,DEVICE=DEVICE,LAMBDA=LAMBDA,GAMMA=GAMMA,ALPHA=ALPHA,NUM_CLASSES=NUM_CLASSES)

    # 모델 저장
    torch.save(model.state_dict(), "segmentation_model_A4_2.pth")
    print("--- Training complete and model saved ---")