import torch
import torch.nn as nn
from torch.amp import autocast,GradScaler
import time

from torchmetrics.classification import MulticlassJaccardIndex
from segmentation_models_pytorch.losses import DiceLoss,FocalLoss


# Early Stopping 
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_validation_loss = float('inf')
        self.counter = 0

    def early_stop(self, validation_loss):
        if validation_loss < self.best_validation_loss - self.min_delta:
            self.best_validation_loss = validation_loss
            self.counter = 0  
        elif validation_loss > self.best_validation_loss + self.min_delta:
            self.counter += 1  
            if self.counter >= self.patience:
                return True
        return False
    
# train model
def train_model(model, train_loader, val_loader,optimizer, num_epochs,patience, DEVICE, LAMBDA, GAMMA, ALPHA, NUM_CLASSES):
    use_cuda_amp = (DEVICE.type == 'cuda')
    scaler = GradScaler() if use_cuda_amp else None

    early_stopper = EarlyStopper(patience=patience)

    ce_criterion = nn.CrossEntropyLoss()
    dice_criterion = DiceLoss(mode='multiclass', classes=NUM_CLASSES)
    #focal_criterion = FocalLoss(mode='multiclass',gamma=GAMMA,alpha=ALPHA,ignore_index=255)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE, dtype=torch.long)

            optimizer.zero_grad()
            if use_cuda_amp:
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss_ce = ce_criterion(outputs, labels)
                    #loss_focal = focal_criterion(outputs,labels)
                    loss_dice = dice_criterion(outputs, labels)
                    loss = (LAMBDA * loss_ce) + ((1 - LAMBDA) * loss_dice) # LossFunction 하이브리드
            
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # MPS/CPU 사용 시: autocast 없이 직접 계산
                outputs = model(images)
                #loss_focal = focal_criterion(outputs,labels)
                loss_ce = ce_criterion(outputs, labels)
                loss_dice = dice_criterion(outputs, labels)
                loss = (LAMBDA * loss_ce) + ((1 - LAMBDA) * loss_dice)

                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        val_loss, val_miou = evaluate_model(model, val_loader, NUM_CLASSES, DEVICE, LAMBDA, GAMMA, ALPHA)
        
        end_time = time.time()
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        print(f"Time: {end_time - start_time:.2f}s")
        print(f"Train Loss: {epoch_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}%")
        print("-" * 30)


        if early_stopper.early_stop(val_loss):
            print(f"Early stopping at epoch {epoch+1}. Validation loss did not improve for {patience} epochs.")
            break  # 학습 루프 중단

    print("Training Finished.")

# evaluate model
def evaluate_model(model, loader,num_classes, DEVICE, LAMBDA, GAMMA, ALPHA):
    model.eval()
    total_loss = 0.0

    ce_criterion = nn.CrossEntropyLoss()
    #focal_criterion = FocalLoss(mode='multiclass',gamma=GAMMA,alpha=ALPHA,ignore_index=255)
    dice_criterion = DiceLoss(mode='multiclass', classes=num_classes)

    metric = MulticlassJaccardIndex(num_classes=num_classes, average='macro').to(DEVICE)
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE, dtype=torch.long)
            
            outputs = model(images)

            loss_ce = ce_criterion(outputs, labels)
            #loss_focal = focal_criterion(outputs,labels)
            loss_dice = dice_criterion(outputs, labels)
            loss = (LAMBDA * loss_ce) + ((1 - LAMBDA) * loss_dice)


            total_loss += loss.item()
            
            metric.update(outputs, labels)
           
    avg_loss = total_loss / len(loader)
    miou = metric.compute().item()
    metric.reset()
    
    return avg_loss, miou