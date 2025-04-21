# Office Objects Detection

Bu proje, ofis ortamında bulunan 8 farklı nesneyi tespit edebilen bir derin öğrenme modeli geliştirmeyi amaçlamaktadır. Model, özel olarak tasarlanmış bir CNN mimarisi kullanarak nesneleri sınıflandırmaktadır.

## Proje Yapısı

```
office_objects_detection/
├── data/                    # Veri seti ve etiketler
│   └── raw_images/         # Ham görüntüler
├── models/                  # Eğitilmiş modeller
├── src/                    # Kaynak kodlar
│   ├── custom_cnn.py       # Özel CNN mimarisi
│   ├── prepare_dataset.py  # Veri hazırlama ve augmentasyon
│   ├── train.py           # Eğitim scripti
│   ├── test.py            # Test ve tahmin scripti
│   ├── visualize_and_predict.py  # Görselleştirme ve tahmin
│   └── download_new_images.py    # Yeni görseller indirme
├── requirements.txt        # Gerekli paketler
└── README.md              # Proje dokümantasyonu
```

## Gereksinimler

- Python 3.8+
- CUDA ve cuDNN (NVIDIA GPU kullanıcıları için)
- requirements.txt'de belirtilen Python paketleri

## Kurulum

1. Sanal ortam oluşturun:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## Veri Seti

Proje, ofis ortamında bulunan 8 farklı nesneyi içeren görüntülerden oluşmaktadır:
- Bilgisayar
- Klavye
- Fare
- Telefon
- Kalem
- Defter
- Bardak
- Çanta

Veri seti yapısı:
```
data/raw_images/
├── bilgisayar/
├── klavye/
├── fare/
├── telefon/
├── kalem/
├── defter/
├── bardak/
└── canta/
```

## Model Mimarisi

Model, aşağıdaki katmanlardan oluşan özel bir CNN mimarisi kullanmaktadır:
- Konvolüsyonel katmanlar (Conv2D)
- Batch Normalization
- ReLU aktivasyon fonksiyonları
- Max Pooling katmanları
- Dropout katmanları
- Dense katmanlar

## Kullanım

### Veri Hazırlama

Veri setini hazırlamak için:
```bash
python src/prepare_dataset.py
```

### Model Eğitimi

Model eğitimi için:
```bash
python src/train.py
```

Eğitim parametreleri:
- Batch size: 32
- Epochs: 50
- Learning rate: 0.001
- Optimizer: Adam

### Test ve Tahmin

1. Test setinde model performansını değerlendirmek için:
```bash
python src/test.py --test
```

2. Tek bir görsel için tahmin yapmak için:
```bash
python src/test.py --image path/to/image.jpg
```

3. Yeni görseller indirmek için:
```bash
python src/download_new_images.py
```

4. Görselleştirme ve tahmin arayüzü için:
```bash
python src/visualize_and_predict.py
```

Arayüzde şu özellikler bulunmaktadır:
- Rastgele görsel seçme ve tahmin yapma
- Bilgisayardan görsel seçme ve tahmin yapma
- Webcam ile gerçek zamanlı tahmin yapma

## Performans Metrikleri

Model performansı aşağıdaki metriklerle değerlendirilir:
- Test doğruluğu (Accuracy)
- Sınıf bazlı doğruluk
- Karışıklık matrisi

## Çıktılar

Eğitim ve test sırasında aşağıdaki çıktılar oluşturulur:
- `models/`: Eğitilmiş model dosyaları (.pth)
- `training_metrics.png`: Eğitim metrikleri grafiği
- `confusion_matrix.png`: Karışıklık matrisi
- `training.log`: Eğitim logları
- `testing.log`: Test logları

## Hata Ayıklama

Yaygın hatalar ve çözümleri:
1. CUDA hatası: NVIDIA sürücülerinin güncel olduğundan emin olun
2. Bellek hatası: Batch size'ı düşürün
3. Veri yükleme hatası: Veri seti yapısını kontrol edin

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın. 

## Youtube video linki



