import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import logging
from datetime import datetime

from custom_cnn import get_model

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_data_loaders(batch_size=32):
    data_dir = "data/processed"
    
    # Veri artırma ve normalizasyon
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Veri setlerini yükle
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=transform
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=transform
    )

    # Sınıf isimlerini ve sayılarını logla
    logging.info(f"Eğitim seti sınıfları: {train_dataset.classes}")
    logging.info(f"Eğitim seti görsel sayısı: {len(train_dataset)}")
    logging.info(f"Doğrulama seti görsel sayısı: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, device, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    best_val_acc = 0.0
    best_model_path = None

    for epoch in range(num_epochs):
        # Eğitim
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Doğrulama
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        # Öğrenme oranını güncelle
        scheduler.step(val_loss)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if best_model_path:
                os.remove(best_model_path)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = f"models/model_{timestamp}_acc{val_acc:.2f}.pth"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'Yeni en iyi model kaydedildi: {best_model_path}')

def main():
    setup_logging()
    
    # CUDA kontrolü
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Eğitim için {device} cihazı kullanılacak")
    
    # Modeli yükle
    model = get_model(device)
    
    # Veri yükleyicileri al
    train_loader, val_loader = get_data_loaders()
    
    # Modeli eğit
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main() 