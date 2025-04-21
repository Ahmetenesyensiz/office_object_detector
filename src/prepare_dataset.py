import os
import shutil
import random
from PIL import Image
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_directory_structure():
    """Eğitim ve doğrulama dizinlerini oluşturur"""
    base_dir = "data/processed"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    # Eski dizinleri temizle
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    # Dizinleri oluştur
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    return train_dir, val_dir

def process_images(source_dir, train_dir, val_dir, val_split=0.2):
    """Görselleri eğitim ve doğrulama setlerine ayırır"""
    # Sınıf dizinlerini al
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) and d != "processed"]
    
    for class_name in classes:
        # Sınıf dizinlerini oluştur
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Görselleri listele
        source_class_dir = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(source_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            logging.warning(f"{class_name} sınıfında görsel bulunamadı!")
            continue
            
        # Görselleri karıştır
        random.shuffle(images)
        
        # Doğrulama seti için görsel sayısını hesapla
        val_size = int(len(images) * val_split)
        
        # Görselleri kopyala
        for i, img_name in enumerate(images):
            source_path = os.path.join(source_class_dir, img_name)
            
            try:
                # Görseli aç ve kontrol et
                with Image.open(source_path) as img:
                    img.verify()
                
                # Doğrulama setine veya eğitim setine kopyala
                if i < val_size:
                    dest_path = os.path.join(val_class_dir, img_name)
                else:
                    dest_path = os.path.join(train_class_dir, img_name)
                
                shutil.copy2(source_path, dest_path)
                logging.info(f"Görsel kopyalandı: {dest_path}")
                
            except Exception as e:
                logging.error(f"Görsel işlenirken hata oluştu: {source_path} - {str(e)}")
        
        logging.info(f"{class_name} sınıfı için {len(images)} görsel işlendi:")
        logging.info(f"  Eğitim seti: {len(images) - val_size} görsel")
        logging.info(f"  Doğrulama seti: {val_size} görsel")

def main():
    setup_logging()
    
    # Dizin yapısını oluştur
    train_dir, val_dir = create_directory_structure()
    
    # Görselleri işle
    process_images("data", train_dir, val_dir)
    
    logging.info("Veri seti hazırlandı!")

if __name__ == "__main__":
    main() 