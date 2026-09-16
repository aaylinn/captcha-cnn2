# -*- coding: utf-8 -*-
"""
PAYLASILAN ON-ISLEME MODULU (noktali/cizgili captcha)
========================================================
captcha_egit.py VE captcha_tahmin.py bu modulden import eder - egitim ve
tahminde farkli on-isleme yapilmasi riski (train/serve skew) ortadan kalkar.

Gradyanli projedeki gibi bir "gradyan duzlestirme" YOK - buradaki bozucu
gorseldeki renkli cizgiler/noktalar, gradyan degil. Cizgi/nokta kaldirma
denemek yerine (gradyanli projede de benzer denemeler is-e yaramamisti,
bkz. proje notlari) CNN'in bu gurultuyu kendi ogrenmesine izin veriyoruz -
CAPTCHA kirma literaturunde de standart yaklasim budur.

Tek on-isleme: griye cevir + sabit boyuta olcekle. Renk bilgisi (metin/cizgi
ayrimi icin ipucu olabilir) BILEREK atiliyor - once gri ile basit model
denenip, ihtiyac olursa RGB varyanti (GIRIS_KANAL=3) denenebilir.
"""

import numpy as np
import cv2

GIRIS_YUKSEKLIK = 64
# DENENDI VE GERI ALINDI: 320'ye (T=40) cikarmak "ardisik ayni karakterlere
# CTC'ye daha fazla alan verir" hipoteziyle denendi, ama SONUC KOTU CIKTI -
# hem genel dogruluk (~%84.70 -> %78.53) hem de hedeflenen uzunluk-hatasi
# orani (~%54 -> %63.5) kotulesti. Fazla T, ayni dizi icin cok fazla gecerli
# CTC hizalama yolu yaratip sinyali seyreltmis olabilir. 256'da (T=32) kalmaya
# devam - bu deger 3 ayri iyilestirme turunun (AdamW, veri artisi, etiket
# temizligi) uzerine kanitlanmis en iyi sonucu veriyor.
GIRIS_GENISLIK = 256


def gorsel_oku(yol):
    """Windows'ta Turkce/unicode yol sorunu icin cv2.imread yerine
    np.fromfile + cv2.imdecode kullanilir (bkz. cptchYeni/veriler toplu
    etiketleme scriptlerinde de aynı sorun cikmisti)."""
    veri = np.fromfile(yol, dtype=np.uint8)
    return cv2.imdecode(veri, cv2.IMREAD_GRAYSCALE)


def on_isle(gri, hedef_h=GIRIS_YUKSEKLIK, hedef_w=GIRIS_GENISLIK):
    """EGITIM ve TAHMIN icin ORTAK on-isleme hatti."""
    return cv2.resize(gri, (hedef_w, hedef_h), interpolation=cv2.INTER_AREA)
