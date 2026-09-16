# -*- coding: utf-8 -*-
"""
PAYLASILAN MODEL TANIMI (PyTorch) - noktali/cizgili captcha
==============================================================
captcha_egit.py VE captcha_tahmin.py bu modulden import eder.

MIMARI: CRNN + CTC
--------------------
Gradyanli projedeki "paylasimli kolon" mimarisi burada KULLANILAMAZ - o
mimari SABIT 5 karakter varsayimina dayaniyordu (oznitelik haritasi tam
5 kolona indiriliyordu). Bu projede karakter sayisi DEGISKEN (3-6 arasi,
olculen veriye gore). Degisken uzunluklu dizi etiketleme problemi icin
standart cozum CTC (Connectionist Temporal Classification) kaybidir:
- Model, girdiyi T zaman-adimlik bir dizi olarak isler (T, karakter
  sayisindan HER ZAMAN buyuk olacak sekilde sabittir, GIRIS_GENISLIK=256
  ile T=32 - bkz onisleme.py; T'yi 40'a cikarmak denendi ama sonuc kotu
  cikti, bkz o dosyadaki not).
- Her zaman-adiminda (alfabe + "bos" sinifi) uzerinde bir olasilik dagilimi
  uretir.
- CTC kaybi, "hangi zaman-adiminin hangi karaktere karsilik geldigini"
  ETIKETLEMEDEN, sadece dogru KARAKTER DIZISININ (5,-,-,x,x,7 gibi butun
  hizalamalarin toplamini) olasiligini maksimize ederek egitilir.
- Tahminde: her adimda en olasi sinif secilir (greedy), ardindan ust uste
  tekrarlar ve "bos" sinifi silinerek nihai metin elde edilir.

Govde: klasik CNN (yukseklik agresif kucultulur, genislik zaman-ekseni
olarak korunur) + BiLSTM (zaman-adimlari arasinda baglam kurar - CTC ile
CRNN'lerde standarttir, bkz. Shi et al. 2015 "CRNN" mimarisi).
"""

import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, sinif_sayisi, gizli_boyut=256):
        """sinif_sayisi: alfabe_uzunlugu + 1 (0. indeks CTC 'bos' sinifi icin ayrilir)."""
        super().__init__()

        def blok(ic, dis, pool=(2, 2)):
            return nn.Sequential(
                nn.Conv2d(ic, dis, 3, padding=1, bias=False),
                nn.BatchNorm2d(dis),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )

        # Giris: (B,1,64,256)
        self.govde = nn.Sequential(
            blok(1, 32, (2, 2)),      # -> (B,32,32,128)
            blok(32, 64, (2, 2)),     # -> (B,64,16,64)
            blok(64, 128, (2, 2)),    # -> (B,128,8,32)
            blok(128, 128, (2, 1)),   # -> (B,128,4,32)  genislik artik sabit (zaman ekseni)
            blok(128, 128, (2, 1)),   # -> (B,128,2,32)
            blok(128, 128, (2, 1)),   # -> (B,128,1,32)  yukseklik 1'e indi
        )

        # NOT: 0.4+0.4 (LSTM ic dropout + cikis dropout) UST USTE cok agresifti -
        # kucuk veri setinde (4-5k ornek) ogrenmeyi neredeyse durduruyordu. 0.3'e
        # cektik: hala augmentation'a karsi bir miktar regularizasyon sagliyor
        # ama modelin ilk etapta veriyi ogrenmesini engellemiyor.
        self.rnn = nn.LSTM(
            input_size=128, hidden_size=gizli_boyut, num_layers=2,
            bidirectional=True, batch_first=True, dropout=0.3,
        )
        self.dropout = nn.Dropout(p=0.3)
        
        self.cikis = nn.Linear(gizli_boyut * 2, sinif_sayisi)

    def forward(self, x):
        """x: (B,1,64,256) -> logits: (T,B,sinif_sayisi) (CTCLoss'un bekledigi sira)."""
        x = self.govde(x)                          # (B,128,1,32)
        b, c, h, w = x.shape
        assert h == 1, f"yukseklik 1'e inmedi: {h} (girdi boyutunu kontrol et)"
        x = x.squeeze(2).permute(0, 2, 1)           # (B,32,128) - zaman ekseni ortada
        x, _ = self.rnn(x)                          # (B,32,gizli_boyut*2)

        # DEĞİŞİKLİK 3: Özellikleri Dropout'tan geçiriyoruz
        x = self.dropout(x)

        x = self.cikis(x)                           # (B,32,sinif_sayisi)
        x = x.permute(1, 0, 2)                      # (T,B,sinif_sayisi) - CTCLoss formati
        return x

def model_kaydet(model, yol, sinif_sayisi):
    torch.save({"model_state": model.state_dict(), "sinif_sayisi": sinif_sayisi}, yol)

def model_yukle(yol, cihaz):
    veri = torch.load(yol, map_location=cihaz, weights_only=False)
    model = CRNN(veri["sinif_sayisi"]).to(cihaz)
    model.load_state_dict(veri["model_state"])
    model.eval()
    return model

def cihaz_sec():
    if torch.cuda.is_available():
        ad = torch.cuda.get_device_name(0)
        print(f"GPU kullanilacak: {ad}")
        return torch.device("cuda")
    print("UYARI: GPU bulunamadi, CPU kullanilacak (yavas olur).")
    return torch.device("cpu")