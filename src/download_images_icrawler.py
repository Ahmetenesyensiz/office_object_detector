import os
import re
import time
import torch
import logging
from icrawler.builtin import GoogleImageCrawler
import unicodedata
from PIL import Image
import shutil
import hashlib

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_download.log'),
        logging.StreamHandler()
    ]
)

def check_cuda():
    """CUDA ve cuDNN kullanılabilirliğini kontrol eder"""
    if torch.cuda.is_available():
        logging.info(f"CUDA kullanılabilir. GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"cuDNN versiyonu: {torch.backends.cudnn.version()}")
        return True
    else:
        logging.warning("CUDA kullanılamıyor. CPU kullanılacak.")
        return False

def remove_turkish_chars(text):
    """Türkçe karakterleri İngilizce karşılıklarına çevirir"""
    tr_chars = {
        'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
        'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U'
    }
    for tr_char, en_char in tr_chars.items():
        text = text.replace(tr_char, en_char)
    return text

def sanitize_filename(filename):
    """Dosya adını güvenli hale getirir"""
    filename = remove_turkish_chars(filename)
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)
    return filename.strip('-')

def calculate_image_hash(image_path):
    """Görselin hash değerini hesaplar"""
    try:
        with Image.open(image_path) as img:
            # Görseli küçük bir boyuta yeniden boyutlandır
            img = img.resize((8, 8), Image.Resampling.LANCZOS)
            # Gri tonlamaya çevir
            img = img.convert('L')
            # Hash değerini hesapla
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            bits = ''.join(['1' if (px >= avg) else '0' for px in pixels])
            return int(bits, 2)
    except Exception as e:
        logging.error(f"Hash hesaplama hatası ({image_path}): {str(e)}")
        return None

def is_duplicate_image(image_path, existing_hashes):
    """Görselin daha önce indirilip indirilmediğini kontrol eder"""
    new_hash = calculate_image_hash(image_path)
    if new_hash is None:
        return True
    return new_hash in existing_hashes

def check_image_quality(image_path, min_size=(200, 200)):
    """Görsel kalitesini kontrol eder"""
    try:
        with Image.open(image_path) as img:
            # Boyut kontrolü
            if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                return False
            
            # Boş veya tek renkli görsel kontrolü
            if img.getextrema() == ((0, 0), (0, 0), (0, 0)):
                return False
            
            return True
    except Exception as e:
        logging.error(f"Görsel kontrolü sırasında hata: {str(e)}")
        return False

def download_with_retry(crawler, keyword, max_num, max_retries=3):
    """Belirli sayıda deneme ile görsel indirme"""
    for attempt in range(max_retries):
        try:
            crawler.crawl(
                keyword=keyword,
                max_num=max_num,
                min_size=(200, 200),
                max_size=None
            )
            return True
        except Exception as e:
            logging.warning(f"Deneme {attempt + 1}/{max_retries} başarısız: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)  # 5 saniye bekle
            else:
                logging.error(f"Tüm denemeler başarısız: {keyword}")
                return False

def main():
    # CUDA kontrolü
    use_cuda = check_cuda()
    
    # Sınıflar ve arama terimleri
    classes = {
        'bilgisayar': ['laptop computer', 'desktop computer', 'office computer'],
        'klavye': ['computer keyboard', 'mechanical keyboard', 'office keyboard'],
        'fare': ['computer mouse', 'wireless mouse', 'office mouse'],
        'telefon': ['office phone', 'desk phone', 'business phone'],
        'kalem': ['office pen', 'ballpoint pen', 'writing pen'],
        'defter': ['notebook', 'office notebook', 'writing notebook'],
        'bardak': ['office mug', 'coffee mug', 'desk mug'],
        'canta': ['office bag', 'laptop bag', 'business bag']
    }

    # Her sınıf için indirilecek görsel sayısı (artırıldı)
    images_per_class = 300  # Her sınıf için 300 görsel

    # Ana veri dizini
    base_dir = 'data'
    os.makedirs(base_dir, exist_ok=True)

    # Her sınıf için görselleri indir
    for class_name, search_terms in classes.items():
        safe_class_name = sanitize_filename(class_name)
        class_dir = os.path.join(base_dir, safe_class_name)
        os.makedirs(class_dir, exist_ok=True)

        logging.info(f"\n[{safe_class_name}] için görseller indiriliyor...")
        
        # Mevcut görsellerin hash değerlerini topla
        existing_hashes = set()
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                img_hash = calculate_image_hash(img_path)
                if img_hash is not None:
                    existing_hashes.add(img_hash)
        
        logging.info(f"Mevcut görsel sayısı: {len(existing_hashes)}")
        
        for search_term in search_terms:
            logging.info(f"  Arama terimi: {search_term}")
            
            # Geçici klasör oluştur
            temp_dir = os.path.join(class_dir, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                crawler = GoogleImageCrawler(
                    storage={'root_dir': temp_dir},
                    feeder_threads=1,
                    parser_threads=1,
                    downloader_threads=4
                )
                
                if download_with_retry(
                    crawler,
                    search_term,
                    images_per_class // len(search_terms)
                ):
                    # İndirilen görselleri kontrol et ve taşı
                    for filename in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, filename)
                        if check_image_quality(file_path) and not is_duplicate_image(file_path, existing_hashes):
                            # Hash değerini ekle
                            img_hash = calculate_image_hash(file_path)
                            if img_hash is not None:
                                existing_hashes.add(img_hash)
                                # Görseli ana klasöre taşı
                                shutil.move(file_path, os.path.join(class_dir, filename))
                            else:
                                os.remove(file_path)
                        else:
                            os.remove(file_path)
            
            except Exception as e:
                logging.error(f"  Hata: {search_term} için görseller indirilemedi - {str(e)}")
                continue
            finally:
                # Geçici klasörü temizle
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        
        logging.info(f"{safe_class_name} için toplam görsel sayısı: {len(os.listdir(class_dir))}")

    logging.info("\n✅ Tüm görseller indirildi.")

if __name__ == "__main__":
    main() 