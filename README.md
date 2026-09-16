# lucamodel — Noktalı/Çizgili CAPTCHA Metin Okuma Modeli

Bu klasör, **luca** adıyla ayrılmış, bağımsız bir CAPTCHA metin okuma
modelidir. Repodaki diğer modellerle karışmaması için ayrı bir klasörde
tutulur — kendi kod kopyalarını, eğitilmiş ağırlıklarını ve sonuç
raporlarını içerir.

## Ne yapıyor

Görsel CAPTCHA'ları (arka planda renkli çizgi/nokta gürültüsü olan,
3-6 karakter uzunluğunda küçük harf + rakam karışımı metin içeren)
otomatik olarak okuyup metne çeviren bir CRNN + CTC modeli.

## Sonuçlar (son eğitim)

| Metrik | Değer |
|---|---|
| Eğitim verisi | 8434 etiketli örnek (7591 eğitim + tekrarlayan-karakter çoğaltmasıyla ~9730, 843 doğrulama) |
| **Doğrulama (tam kelime) doğruluğu** | **%90.75** |
| Doğrulama karakter hata oranı (CER) | %2.10 (karakterlerin ~%97.9'u doğru) |
| Eğitim (train) doğruluğu | %97.87 |

Detaylı log: [`egitim_raporu.txt`](egitim_raporu.txt), eğrileri:
[`egitim_gecmisi.png`](egitim_gecmisi.png), doğrulama setindeki kalan
hatalar (elle inceleme için): [`val_hatalari.txt`](val_hatalari.txt).

## Mimari

**CRNN + CTC** (Connectionist Temporal Classification):
- CNN gövdesi görseli (64×256 gri ton) 32 zaman-adımlık bir diziye indirger
- 2 katmanlı çift-yönlü LSTM zaman-adımları arasında bağlam kurar
- CTC kaybı, karakter sayısı değişken (3-6) olduğu için sabit hizalama
  gerektirmeden dizi öğrenimini mümkün kılar

Karakter sayısı SABİT olmadığı için (gradyanlı/farklı bir projedeki
"sabit 5 kolon" mimarisinin aksine) bu yaklaşım seçildi. Detay için
[`captcha_model.py`](captcha_model.py) içindeki modül docstring'ine bakın.

## Eğitim sürecinde önemli kararlar

- **AdamW + weight_decay=1e-4**: düz `Adam` yerine, eğitim/doğrulama
  arasındaki ezberleme farkını azaltmak için
- **Dropout 0.3** (LSTM içi + çıkış öncesi) — çok yüksek dropout (0.4+0.4)
  denenmiş ama öğrenmeyi neredeyse durdurmuştu, 0.3'e çekildi
- **Hafif veri artırma (augmentation)**: sadece eğitim setine, ±5°
  döndürme + %5 kaydırma/ölçekleme (`captcha_egit.py` içindeki
  `CaptchaVeriSeti`)
- **Tekrarlayan-karakter çoğaltması**: hata analizinde en büyük hata
  kategorisinin (~%54) ardışık aynı karakterleri (`66`, `mm` gibi) CTC'nin
  tek karakter sayıp yutması olduğu görüldü. Çözüm: bu deseni içeren
  gerçek eğitim örnekleri 3x fazladan train setine eklendi (yeni etiketleme
  gerekmedi, mevcut veriden). Bu değişiklik hem hedef sorunu (%54→%32)
  hem de genel doğrulama doğruluğunu (%84.70→%90.75) belirgin şekilde
  iyileştirdi.
- **Girdi çözünürlüğünü artırma (256→320 genişlik) DENENDİ VE GERİ ALINDI**
  — beklentinin aksine sonucu kötüleştirdi (bkz. `onisleme.py` içindeki not).
- **Erken durdurma + otomatik model yedekleme**: her yeni eğitim, üzerine
  yazmadan önce mevcut en iyi modeli otomatik yedekler ve yeni sonucu
  eskisiyle karşılaştırır (bkz. `captcha_egit.py`).

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `lucamodel.pt` | Eğitilmiş model ağırlıkları (en iyi checkpoint) |
| `alfabe.json` | Kullanılan alfabe (35 karakter) + ön-işleme ayarları |
| `captcha_model.py` | Model mimarisi (CRNN) + kaydet/yükle yardımcıları |
| `captcha_egit.py` | Eğitim betiği (veri okuma, döngü, rapor üretimi) |
| `onisleme.py` | Ortak ön-işleme (griye çevirme + boyutlandırma) — eğitim ve tahmin ile aynı olmalı |
| `tahmin_et.py` | **Bağımsız, hazır çalışan tahmin scripti** — `python tahmin_et.py resim.png` |
| `egitim_raporu.txt` | Son eğitimin özet sonucu |
| `egitim_gecmisi.png` | Kayıp/doğruluk eğrileri |
| `val_hatalari.txt` | Doğrulama setinde hâlâ yanlış tahmin edilen örnekler |
| `veri_etiketleri.csv` | Tüm eğitim verisinin dosya adı ↔ doğru etiket eşlemesi (8434 satır) — Roboflow'a yüklenen görsellerin yedek referansı |
| `araclar/supheli_kontrol.py` | Şüpheli/düşük-güvenli etiketleri tek tek gözden geçirme arayüzü (Flask) |
| `araclar/yeni_veri_etiketle.py` | Model yardımlı, güven-eşikli otomatik+manuel etiketleme arayüzü (Flask) |

**Not:** `captcha_egit.py` ve `araclar/` içindeki araçlar, orijinal proje
yapısında `cptchYeni/veriler/` klasörüne (ham görseller + `labels_lower.csv`)
göreli yol kullanıyor. Bu klasördeki kod kopyaları referans/dokümantasyon
amaçlıdır — çalıştırmak için kendi veri klasör yapınızı buna göre ayarlamanız
gerekir (ham CAPTCHA görselleri boyut/gizlilik nedeniyle bu repoya dahil
edilmedi).

## Kullanım (tahmin için)

Bağımsız, hazır çalışan bir script var — sadece bu klasördeki dosyalara
ihtiyaç duyar, başka bir yere bağımlı değil:

```bash
pip install -r requirements.txt
python tahmin_et.py yol/to/captcha.png
```

Çıktı: `Tahmin: ufvq` gibi düz metin. Kod içinden çağırmak istersen:

```python
from tahmin_et import tahmin_et

metin = tahmin_et("captcha.png")
print(metin)
```

(`tahmin_et.py` içindeki `greedy_decode` fonksiyonu, model çıktısını
—T zaman-adımlık olasılık dizisini— nihai metne çeviriyor: her adımda en
olası sınıf seçilir, ardışık tekrarlar ve CTC "boş" sınıfı silinir.)
