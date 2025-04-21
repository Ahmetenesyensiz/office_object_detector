import os
from PIL import Image
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_directory(directory):
    """Dizindeki görselleri kontrol eder"""
    if not os.path.exists(directory):
        logging.error(f"Dizin bulunamadı: {directory}")
        return 0
    
    image_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                    image_count += 1
                except Exception as e:
                    logging.error(f"Geçersiz görsel: {image_path} - {str(e)}")
    
    return image_count

def main():
    setup_logging()
    
    # Eğitim ve doğrulama dizinlerini kontrol et
    train_dir = "data/processed/train"
    val_dir = "data/processed/val"
    
    # Sınıf sayısını kontrol et
    train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    val_classes = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    
    logging.info(f"Eğitim seti sınıfları: {train_classes}")
    logging.info(f"Doğrulama seti sınıfları: {val_classes}")
    
    # Görsel sayılarını kontrol et
    train_count = check_directory(train_dir)
    val_count = check_directory(val_dir)
    
    logging.info(f"Eğitim seti görsel sayısı: {train_count}")
    logging.info(f"Doğrulama seti görsel sayısı: {val_count}")
    
    # Sınıf başına görsel sayılarını kontrol et
    for class_name in train_classes:
        class_dir = os.path.join(train_dir, class_name)
        count = check_directory(class_dir)
        logging.info(f"{class_name} sınıfı eğitim görsel sayısı: {count}")
    
    for class_name in val_classes:
        class_dir = os.path.join(val_dir, class_name)
        count = check_directory(class_dir)
        logging.info(f"{class_name} sınıfı doğrulama görsel sayısı: {count}")

if __name__ == "__main__":
    main() 