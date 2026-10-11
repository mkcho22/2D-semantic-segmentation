import torch
import matplotlib.pyplot as plt 
import numpy as np
from torchvision import transforms
from utils import UNet
from data import CarlaDataset
from torch.utils.data import DataLoader
# 모델 로딩
print("Loading model structure...")

try:
    from utils import UNet
except ImportError:
    print("오류: utils.py 또는 UNet 클래스를 찾을 수 없습니다.")

NUM_CLASSES = 13
MODEL_PATH = "segmentation_model_A4_2.pth"
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

model = UNet(n_classes=NUM_CLASSES)

print(f"Loading trained weights from {MODEL_PATH}...")

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model = model.to(DEVICE)
model.eval()


print("Loading validation dataset...")

DATA_DIR = "./2D Segmentation"
INPUT_HEIGHT = 600
INPUT_WIDTH = 800
BATCH_SIZE = 4 

image_transform = transforms.Compose([
    transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
label_transform = transforms.Compose([
    transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH), interpolation=transforms.InterpolationMode.NEAREST)
])

val_dataset = CarlaDataset(root_dir=DATA_DIR, split="Validation", 
                           image_transform=image_transform, 
                           label_transform=label_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 결과 시각화
#--------------------------------------------------------------------------
print("\n--- Visualizing Segmentation Results ---")

CLASS_COLORS = np.array([
    [0, 0, 0],         # 0: None (검정) 
    [70, 70, 70],      # 1: Buildings (어두운 회색)
    [190, 153, 153],   # 2: Fences (연한 갈색)
    [55, 90, 80],      # 3: Other (어두운 청록색)
    [220, 20, 60],     # 4: Pedestrians (진홍색)    
    [153, 153, 153],   # 5: Poles (중간 회색)
    [157, 234, 50],    # 6: RoadLines (밝은 연두색)
    [128, 64, 128],    # 7: Roads (자주색, 도로 면)
    [244, 35, 232],    # 8: Sidewalks (핑크)
    [107, 142, 35],    # 9: Vegetation (올리브 녹색)
    [220, 220, 0],     # 10: Vehicles (노랑)
    [102, 102, 156],   # 11: Walls (회색 보라)
    [220, 200, 0]      # 12: TrafficSigns (진한 노란색)
], dtype=np.uint8)

# 시각화할 이미지 개수
NUM_VISUALIZE = 5

data_iter = iter(val_loader) 

plt.figure(figsize=(NUM_VISUALIZE * 4, 10)) 

for i in range(NUM_VISUALIZE):
    try:
        images, true_labels = next(data_iter) 
    except StopIteration:
        print("Reached end of validation dataset.")
        break
    
    image = images[0].cpu().numpy().transpose(1, 2, 0) 
    true_label = true_labels[0].cpu().numpy()          
    
    unique_classes = np.unique(true_label)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1) 

    # 모델 예측
    model.eval() # 평가 모드
    with torch.no_grad():
        output = model(images.to(DEVICE))
    predicted_label = torch.argmax(output[0], dim=0).cpu().numpy() 

    predicted_colored_mask = CLASS_COLORS[predicted_label]
    true_colored_mask = CLASS_COLORS[true_label]
  
    # 결과 출력
    plt.subplot(3, NUM_VISUALIZE, i + 1)
    plt.imshow(image)
    plt.title(f"Original {i+1}")
    plt.axis('off')

    plt.subplot(3, NUM_VISUALIZE, i + NUM_VISUALIZE + 1)
    plt.imshow(predicted_colored_mask)
    plt.title(f"Predicted Overlay {i+1}")
    plt.axis('off')
    
    plt.subplot(3, NUM_VISUALIZE, i + NUM_VISUALIZE * 2 + 1)
    plt.imshow(true_colored_mask)
    plt.title(f"Ground Truth Overlay {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()

plt.savefig('output_A4_2.png') 
plt.close() # 메모리 해제
print("--- Visualization complete ---")