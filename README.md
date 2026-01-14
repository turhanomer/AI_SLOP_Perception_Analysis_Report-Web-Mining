# YouTube Duygu ve Algı Analizi Projesi

Bu proje, YouTube video yorumlarını analiz ederek tüketici duygu durumunu, algılarını ve endişelerini tespit eden bir web madenciliği ve duygu analizi sistemidir.

## Özellikler

- **Sentiment Analysis**: VADER Sentiment ile duygu analizi
- **Topic Modeling**: LDA ile negatif yorumlarda ana konu tespiti
- **Contextual Analysis**: Kriz vs Fırsat kelime analizi
- **Görselleştirme**: Profesyonel grafikler ve kelime bulutları
- **Rapor Oluşturma**: Detaylı analiz raporları (Markdown ve JSON)

## Kurulum

1. **Gereksinimleri yükleyin:**
```bash
pip install -r requirements.txt
```

2. **Konfigürasyon dosyasını oluşturun:**
```bash
cp config.example.py config.py
```

3. **config.py dosyasını düzenleyin:**
   - `YOUTUBE_API_KEY`: YouTube Data API v3 anahtarınızı girin
   - `YOUTUBE_VIDEO_ID`: Analiz edilecek video ID'sini girin

## Kullanım

### Veri Çekme (Opsiyonel)
Eğer yeni veri çekmek istiyorsanız, `main.py` dosyasındaki `fetch_youtube_comments` fonksiyonunu kullanabilirsiniz.

### Analiz Çalıştırma
```bash
python main.py
```

Bu komut:
- CSV dosyasından yorumları yükler
- Sentiment analizi yapar
- Topic modeling uygular
- `analysis_results.json` dosyası oluşturur

### Görselleştirme
```bash
python visualize.py
```

Bu komut `charts/` klasörüne grafikleri kaydeder:
- `sentiment_distribution.png` - Duygu dağılımı
- `crisis_vs_opportunity.png` - Kriz vs Fırsat karşılaştırması
- `top_fears.png` - Tüketici korkuları
- `negative_wordcloud.png` - Negatif kelime bulutu

## Dosya Yapısı

```
Web Mining/
├── main.py                  # Ana analiz scripti
├── visualize.py             # Görselleştirme modülü
├── config.example.py        # Konfigürasyon örneği
├── config.py                # Gerçek konfigürasyon (gitignore'da)
├── requirements.txt         # Python bağımlılıkları
├── .gitignore              # Git ignore dosyası
├── README.md               # Bu dosya
├── charts/                 # Grafikler (oluşturulur)
│   ├── sentiment_distribution.png
│   ├── crisis_vs_opportunity.png
│   ├── top_fears.png
│   └── negative_wordcloud.png
└── WEB_MADENCILIGI_FINAL_RAPORU.md  # Final rapor
```

## Güvenlik

⚠️ **ÖNEMLİ**: 
- `config.py` dosyası `.gitignore`'da olduğu için GitHub'a yüklenmez
- API key'lerinizi asla kod içine yazmayın
- `config.example.py` dosyasını kopyalayıp `config.py` olarak kaydedin ve gerçek değerleri girin

## Lisans

Bu proje eğitim amaçlıdır.
