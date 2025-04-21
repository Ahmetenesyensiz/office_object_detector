import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from custom_cnn import CustomCNN

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testing.log'),
        logging.StreamHandler()
    ]
)

class OfficeClassifier(nn.Module):
    def __init__(self, num_classes):
        super(OfficeClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class OfficeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.images.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_latest_model():
    """
    En son kaydedilen model dosyasını bulur.
    
    Returns:
        str: En son model dosyasının yolu
        
    Raises:
        FileNotFoundError: Model dizini veya model dosyası bulunamazsa
    """
    models_dir = 'models'
    if not os.path.exists(models_dir):
        raise FileNotFoundError("Models directory does not exist. Please train the model first.")
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError("No model files found. Please train the model first.")
    
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
    return os.path.join(models_dir, latest_model)

def evaluate_model(model, test_loader, device):
    """
    Modeli test setinde değerlendirir.
    
    Args:
        model (nn.Module): Değerlendirilecek model
        test_loader (DataLoader): Test veri yükleyici
        device (torch.device): Hesaplama cihazı
        
    Returns:
        float: Test doğruluğu
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 8
    class_total = [0] * 8
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    accuracy = 100 * correct / total
    logging.info(f'Test Accuracy: {accuracy:.2f}%')
    
    for i in range(8):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            logging.info(f'Class {i} Accuracy: {class_accuracy:.2f}%')
    
    return accuracy

def predict_single_image(model, image_path, device, transform, class_names):
    """
    Tek bir görsel için tahmin yapar.
    
    Args:
        model (nn.Module): Eğitilmiş model
        image_path (str): Görsel dosya yolu
        device (torch.device): Hesaplama cihazı
        transform: Görsel dönüşümleri
        class_names (list): Sınıf isimleri listesi
        
    Returns:
        tuple: (tahmin edilen sınıf indeksi, sınıf olasılıkları)
        
    Raises:
        FileNotFoundError: Görsel dosyası bulunamazsa
        ValueError: Görsel yüklenemezse
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")
    
    model.eval()
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    predicted_class = predicted.item()
    class_probs = probabilities[0].cpu().numpy()
    
    # Sonuçları göster
    print(f"\nGörsel: {image_path}")
    print(f"Tahmin edilen sınıf: {class_names[predicted_class]}")
    print("Sınıf olasılıkları:")
    for i, prob in enumerate(class_probs):
        print(f"{class_names[i]}: {prob:.2%}")
    
    return predicted_class, class_probs

def plot_confusion_matrix(model, test_loader, device, class_names):
    """
    Karışıklık matrisini çizer ve kaydeder.
    
    Args:
        model (nn.Module): Eğitilmiş model
        test_loader (DataLoader): Test veri yükleyici
        device (torch.device): Hesaplama cihazı
        class_names (list): Sınıf isimleri listesi
    """
    model.eval()
    confusion_matrix = np.zeros((8, 8), dtype=np.int32)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def main():
    parser = argparse.ArgumentParser(description='Model test ve tahmin işlemleri')
    parser.add_argument('--image', type=str, help='Tahmin yapılacak görsel yolu')
    parser.add_argument('--test', action='store_true', help='Test setinde değerlendirme yap')
    args = parser.parse_args()
    
    # Test veri seti için transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test veri setini yükle
    test_dataset = OfficeDataset('data/processed/test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Modeli yükle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomCNN(num_classes=len(test_dataset.classes))
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # Test et
    preds, labels = test_model(model, test_loader, device)
    
    # Metrikleri hesapla
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    
    logging.info(f'Test Accuracy: {accuracy:.4f}')
    logging.info(f'Test Precision: {precision:.4f}')
    logging.info(f'Test Recall: {recall:.4f}')
    logging.info(f'Test F1 Score: {f1:.4f}')
    
    # Karışıklık matrisini çiz
    plot_confusion_matrix(model, test_loader, device, test_dataset.classes)
    
    if args.image:
        predict_single_image(model, args.image, device, test_transform, test_dataset.classes)

if __name__ == '__main__':
    main() 