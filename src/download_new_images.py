from icrawler.builtin import GoogleImageCrawler
import os
import random
import time

# İndirilecek sınıflar
CLASSES = ['bardak', 'bilgisayar', 'canta', 'defter', 'fare', 'kalem', 'klavye', 'telefon']

# Her sınıf için indirilecek görsel sayısı
IMAGES_PER_CLASS = 10

# İndirilecek görsellerin kaydedileceği klasör
OUTPUT_DIR = 'data/new_test_images'

def download_images():
    # Çıktı klasörünü oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Her sınıf için görsel indir
    for class_name in CLASSES:
        print(f"\n{class_name} sınıfı için görseller indiriliyor...")
        
        # Google arama sorgusu oluştur
        search_query = f"{class_name} ofis nesnesi"
        
        # Google görsel indirici oluştur
        google_crawler = GoogleImageCrawler(
            storage={'root_dir': OUTPUT_DIR},
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4
        )
        
        # Görselleri indir
        google_crawler.crawl(
            keyword=search_query,
            max_num=IMAGES_PER_CLASS,
            min_size=(200, 200),
            max_size=None,
            file_idx_offset=0
        )
        
        # Google'ın rate limit'ini aşmamak için bekle
        time.sleep(2)
    
    print("\nTüm görseller indirildi!")

if __name__ == '__main__':
    download_images() 