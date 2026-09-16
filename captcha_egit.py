# -*- coding: utf-8 -*-
"""
NOKTALI/CIZGILI CAPTCHA - CRNN + CTC EGITIMI
================================================
Veri: cptchYeni/veriler/labels_lower.csv (dosya adi -> etiket, hepsi kucuk
harf - buyuk harf orijinal veride hic yok, elle etiketleme sirasinda font
stilinden kaynakli yanlis okumalar duzeltildi, bkz. labels_lower.csv).

Karakter sayisi DEGISKEN (olculen veriye gore 3-6 arasi) - bu yuzden
gradyanli projedeki "sabit 5 kolon" mimarisi yerine CRNN+CTC kullanilir
(detay: captcha_model.py).

Calistirma:
    python captcha_egit.py

Ciktilar (bu klasorde):
    model_en_iyi.pt      - en iyi (dogrulama kelime-dogrulugu en yuksek) agirliklar
    alfabe.json           - alfabe + on-isleme ayarlari (tahminde de kullanilir)
    egitim_gecmisi.png    - kayip/dogruluk egrileri
    egitim_raporu.txt      - ozet sonuc
    val_hatalari.txt       - dogrulama setinde YANLIS tahmin edilenler (elle inceleme icin)
"""

import os
import json
import random
import csv
import re
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# DEĞİŞİKLİK 1: Veri artırımı (Augmentation) için torchvision eklendi
import torchvision.transforms as T

import onisleme as oi
from captcha_model import CRNN, model_kaydet, cihaz_sec

# ============================ AYARLAR ============================
KLASOR = os.path.dirname(os.path.abspath(__file__))
VERI_DIZIN = os.path.join(KLASOR, "cptchYeni", "veriler")
LABELS_CSV = os.path.join(VERI_DIZIN, "labels_lower.csv")

BATCH_BOYUTU = 32       # 8'e dusurulmustu - BatchNorm icin cok kucuktu (istatistikler
                        # gurultulu oluyordu), 32'ye geri alindi
EPOCH_SAYISI = 90       # augmentation isi biraz zorlastirdigi icin daha fazla epoch
OGRENME_ORANI = 1e-3
AGIRLIK_AZALMASI = 1e-4  # AdamW L2 regularizasyonu - egitim(%96.8)/dogrulama(%79) arasindaki
                         # sabit ~18 puanlik farki (ezberleme) kapatmaya yardimci olmasi icin
                         # eklendi. Dropout'tan daha yumusak/ongorulebilir bir regularizasyon -
                         # LR scheduler'i bozup ogrenmeyi durdurma riski yok.
VAL_ORANI = 0.1          # verinin %10'u dogrulama icin ayrilir
SEED = 42

# val_hatalari.txt analizinde hatalarin ~%54'u "uzunluk hatasi" (ardisik AYNI
# karakterleri CTC'nin tek karakter sayip yutmasi, orn. 666b -> 66b) idi.
# Cozunurluk artirmak bunu COZMEDI (aksine kotulestirdi, bkz. onisleme.py
# notu). Bunun yerine: ardisik tekrarlayan karakter iceren GERCEK egitim
# ornekleri (yeni etiketleme gerektirmez, CSV'de zaten var) train setine
# COGALTMA_KATSAYISI kadar ekstra kopyalanir - her kopya augmentation
# sayesinde farkli bir varyasyonla gorunur, model bu deseni normalden daha
# sik gorur. Sadece TRAIN setine uygulanir, val temiz/temsili kalmali.
TEKRARLAYAN_KARAKTER_COGALTMA = 3
# ===================================================================


def tekrarlayan_karakter_var(etiket):
    return any(etiket[i] == etiket[i + 1] for i in range(len(etiket) - 1))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def levenshtein(a, b):
    """Duzenleme mesafesi (karakter hata orani icin) - ekstra kutuphane gerekmez."""
    if a == b:
        return 0
    m, n = len(a), len(b)
    onceki = list(range(n + 1))
    for i in range(1, m + 1):
        simdi = [i] + [0] * n
        for j in range(1, n + 1):
            maliyet = 0 if a[i - 1] == b[j - 1] else 1
            simdi[j] = min(onceki[j] + 1, simdi[j - 1] + 1, onceki[j - 1] + maliyet)
        onceki = simdi
    return onceki[n]


# ============================ VERI ============================

def veriyi_oku():
    """labels_lower.csv icindeki (dosya, etiket) ciftlerini okur.

    unsure=1 (henuz elle onaylanmamis/supheli) satirlar ATLANIR - bunlar
    supheli_kontrol.py ile gozden gecirilene kadar egitime YANLIS etiket
    olarak karisip CTC kaybini bozabilir. Su an hepsi 0 gorunse de (hepsi
    incelendi), ileride yeni toplu-etiketleme turlarinda tekrar unsure=1
    satirlar eklenirse egitim otomatik olarak onlari disarida birakir."""
    kayitlar = []
    atlanan = 0
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for satir in csv.DictReader(f):
            if satir.get("unsure") == "1":
                atlanan += 1
                continue
            kayitlar.append((satir["filename"], satir["label"]))
    if atlanan:
        print(f"UYARI: {atlanan} satir unsure=1 oldugu icin egitime dahil edilmedi "
              f"(once supheli_kontrol.py ile onayla).")
    return kayitlar


class CaptchaVeriSeti(Dataset):
    # DEĞİŞİKLİK 2: Sadece eğitim setine uygulamak için is_train parametresi eklendi
    def __init__(self, kayitlar, karakter_index, is_train=False):
        self.kayitlar = kayitlar
        self.karakter_index = karakter_index
        self.is_train = is_train
        
        # Eğitim verisi için hafif döndürme ve kaydırma işlemleri.
        # NOT: fill=1.0 (beyaz) verdik - varsayilan fill=0 (siyah) donduruldugunde/
        # kaydirildiginda kenarlara gercek veride hic olmayan siyah bloklar
        # ekliyordu; bu da egitim/dogrulama arasinda kucuk bir dagilim
        # farkina yol aciyordu.
        self.transform = T.Compose([
            T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=1.0)
        ])

    def __len__(self):
        return len(self.kayitlar)

    def __getitem__(self, i):
        dosya, etiket = self.kayitlar[i]
        yol = os.path.join(VERI_DIZIN, dosya)
        gri = oi.gorsel_oku(yol)
        gri = oi.on_isle(gri)
        gorsel = torch.from_numpy((gri.astype(np.float32) / 255.0)[None, :, :])
        
        # DEĞİŞİKLİK 3: Eğitim setindeysek veri artırımını (transform) uygula
        if self.is_train:
            gorsel = self.transform(gorsel)
            
        hedef = [self.karakter_index[c] for c in etiket]
        return gorsel, torch.tensor(hedef, dtype=torch.long), etiket, dosya


def collate_fn(grup):
    """CTCLoss'un bekledigi format: hedefler tek bir duz tensora birlestirilir,
    her ornegin uzunlugu ayri bir tensorda tutulur."""
    gorseller, hedefler, etiketler, dosyalar = zip(*grup)
    gorseller = torch.stack(gorseller)
    uzunluklar = torch.tensor([len(h) for h in hedefler], dtype=torch.long)
    hedefler_duz = torch.cat(hedefler)
    return gorseller, hedefler_duz, uzunluklar, etiketler, dosyalar


# ============================ CTC DECODE ============================

def greedy_decode(logits, ters_index):
    """logits: (T,B,sinif_sayisi) -> her ornek icin metin listesi.
    Greedy: her adimda en olasi sinif, ardindan ardisik tekrarlar + 'bos' (0) silinir."""
    tahminler = logits.argmax(-1).permute(1, 0).cpu().numpy()  # (B,T)
    sonuclar = []
    for dizi in tahminler:
        metin = []
        onceki = -1
        for k in dizi:
            if k != 0 and k != onceki:
                metin.append(ters_index[k])
            onceki = k
        sonuclar.append("".join(metin))
    return sonuclar


# ============================ ONCEKI MODELI KORUMA ============================

def onceki_en_iyi_modeli_yedekle():
    """model_en_iyi.pt (ve iliskili dosyalar) UZERINE YAZILMADAN once otomatik
    yedekler - yeni egitim daha kotu cikarsa bir onceki (iyi) model kaybolmasin.
    Onceki calistirmanin en iyi val dogrulugunu (varsa) okuyup dondurur, boylece
    egitim sonunda 'yeni model eskisinden iyi mi kotu mu' diye karsilastirabiliriz."""
    model_yolu = os.path.join(KLASOR, "model_en_iyi.pt")
    if not os.path.exists(model_yolu):
        return None  # ilk calistirma, yedeklenecek bir sey yok

    zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
    yedek_dizin = os.path.join(KLASOR, "model_yedekler", f"otomatik_{zaman_damgasi}")
    os.makedirs(yedek_dizin, exist_ok=True)
    for dosya in ["model_en_iyi.pt", "alfabe.json", "egitim_raporu.txt",
                  "egitim_gecmisi.png", "val_hatalari.txt"]:
        kaynak = os.path.join(KLASOR, dosya)
        if os.path.exists(kaynak):
            shutil.copy(kaynak, os.path.join(yedek_dizin, dosya))
    print(f"Onceki model otomatik yedeklendi: {yedek_dizin}")

    onceki_dogruluk = None
    onceki_rapor = os.path.join(yedek_dizin, "egitim_raporu.txt")
    if os.path.exists(onceki_rapor):
        with open(onceki_rapor, encoding="utf-8") as f:
            for satir in f:
                if "En iyi dogrulama" in satir:
                    eslesme = re.search(r"%([\d.]+)", satir)
                    if eslesme:
                        onceki_dogruluk = float(eslesme.group(1)) / 100
    return onceki_dogruluk


# ============================ EGITIM ============================

def main():
    onceki_dogruluk = onceki_en_iyi_modeli_yedekle()
    cihaz = cihaz_sec()

    kayitlar = veriyi_oku()
    print(f"Toplam etiketli ornek: {len(kayitlar)}")

    # Alfabe dogrudan veriden cikarilir - elle sabitlemek yerine, veri
    # buyudukce (yeni parti etiketlendikce) otomatik guncel kalir.
    tum_karakterler = sorted(set("".join(e for _, e in kayitlar)))
    karakter_index = {c: i + 1 for i, c in enumerate(tum_karakterler)}  # 0 = CTC 'bos'
    ters_index = {i + 1: c for i, c in enumerate(tum_karakterler)}
    sinif_sayisi = len(tum_karakterler) + 1
    print(f"Alfabe ({len(tum_karakterler)} karakter): {''.join(tum_karakterler)}")

    random.shuffle(kayitlar)
    val_sayisi = max(1, int(len(kayitlar) * VAL_ORANI))
    val_kayitlar = kayitlar[:val_sayisi]
    train_kayitlar = kayitlar[val_sayisi:]
    print(f"Egitim: {len(train_kayitlar)}  Dogrulama: {len(val_kayitlar)}")

    # tekrarlayan karakterli (aa, 66, mm gibi) train ornekleri ekstra kopyalanir
    tekrarlayanlar = [(d, e) for d, e in train_kayitlar if tekrarlayan_karakter_var(e)]
    train_kayitlar = train_kayitlar + tekrarlayanlar * TEKRARLAYAN_KARAKTER_COGALTMA
    random.shuffle(train_kayitlar)
    print(f"Tekrarlayan karakterli {len(tekrarlayanlar)} ornek "
          f"{TEKRARLAYAN_KARAKTER_COGALTMA}x cogaltildi -> "
          f"yeni egitim seti: {len(train_kayitlar)}")

    # DEĞİŞİKLİK 4: is_train parametreleri eklendi
    train_set = CaptchaVeriSeti(train_kayitlar, karakter_index, is_train=True)
    val_set = CaptchaVeriSeti(val_kayitlar, karakter_index, is_train=False)
    
    # NOT: num_workers=0 iken gorsel okuma/resize/augmentation CPU'da TEK
    # is parcaciginda, GPU'yu bekleterek yapiliyordu (yavasligin asil sebebi
    # muhtemelen buydu). num_workers>0 ile PyTorch bu isi arka planda birden
    # fazla surecte, bir sonraki batch'i ONCEDEN hazirlayarak yapar - GPU
    # daha az bekler. persistent_workers=True ile de bu surecler epoch'lar
    # arasi yeniden baslatilmaz (baslatma maliyeti tekrarlanmaz).
    VERI_YUKLEME_SURECI = 4
    train_yukleyici = DataLoader(train_set, batch_size=BATCH_BOYUTU, shuffle=True,
                                  collate_fn=collate_fn, num_workers=VERI_YUKLEME_SURECI,
                                  pin_memory=True, persistent_workers=True)
    val_yukleyici = DataLoader(val_set, batch_size=BATCH_BOYUTU, shuffle=False,
                                collate_fn=collate_fn, num_workers=VERI_YUKLEME_SURECI,
                                pin_memory=True, persistent_workers=True)

    model = CRNN(sinif_sayisi).to(cihaz)
    kayip_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    # Adam -> AdamW: weight_decay eklemek icin (Adam'da weight_decay L2 cezasini
    # yanlis/tutarsiz uygular - AdamW bunu duzeltilmis/ayri sekilde yapar, bu
    # yuzden weight_decay kullanilacaksa AdamW standarttir).
    optimizer = torch.optim.AdamW(model.parameters(), lr=OGRENME_ORANI,
                                   weight_decay=AGIRLIK_AZALMASI)
    # NOT: onceden val_kelime_dogruluk (tam-esleme) izleniyordu - bu metrik
    # egitimin ilk (uzun) evresinde neredeyse hep 0.0 kalir (kisa bir dizide
    # TUM karakterlerin dogru olmasi gerekiyor), bu yuzden scheduler "gelisme
    # yok" saniyor ve LR'yi cok erken/cok sik yariya indiriyordu - modelin
    # daha ogrenecek zamani olmadan LR'nin cokmesine yol aciyordu. Bunun
    # yerine surekli/yumusak degisen val_kayip (loss) izleniyor, patience de
    # artirildi.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=0.5, patience=6)

    # ERKEN DURDURMA: en iyi val dogrulugu ERKEN_DURDURMA_SABRI epoch boyunca
    # hic iyilesmezse egitim EPOCH_SAYISI'na varmadan durur - bir onceki
    # eGitimde son ~7 epoch zaten platoydu (73'te en iyi, 80'e kadar sabit
    # kaldi). Sabit bir epoch sayisi tahmin etmek yerine kod kendi karar
    # versin, hem zaman kazandirir hem de gercekten iyilesme durana kadar
    # devam etmesini garantiler (cok erken kesmez).
    ERKEN_DURDURMA_SABRI = 20
    son_iyilesmeden_beri = 0

    gecmis = {"epoch": [], "train_kayip": [], "val_kayip": [],
              "train_kelime_dogruluk": [], "val_kelime_dogruluk": [],
              "val_karakter_hata_orani": []}
    en_iyi_dogruluk = -1.0
    en_iyi_epoch = 0

    for epoch in range(1, EPOCH_SAYISI + 1):
        # --- egitim ---
        model.train()
        toplam_kayip = 0.0
        train_dogru = 0
        for gorseller, hedefler_duz, uzunluklar, etiketler, _ in train_yukleyici:
            gorseller = gorseller.to(cihaz)
            hedefler_duz = hedefler_duz.to(cihaz)

            logits = model(gorseller)  # (T,B,sinif_sayisi)
            log_olasilik = logits.log_softmax(2)
            T, B, _ = log_olasilik.shape
            girdi_uzunluklari = torch.full((B,), T, dtype=torch.long)

            kayip = kayip_fn(log_olasilik, hedefler_duz, girdi_uzunluklari, uzunluklar)
            optimizer.zero_grad()
            kayip.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            toplam_kayip += kayip.item() * B

            # egitim doguruluk takibi (ekstra forward pass gerekmez, zaten
            # hesaplanmis logit'ler uzerinden decode ediliyor - dropout ACIK
            # oldugu icin val dogrulugundan biraz daha dusuk cikmasi normal)
            with torch.no_grad():
                tahminler = greedy_decode(logits, ters_index)
                train_dogru += sum(1 for t, g in zip(tahminler, etiketler) if t == g)

        train_kayip = toplam_kayip / len(train_set)
        train_dogruluk = train_dogru / len(train_set)

        # --- dogrulama ---
        model.eval()
        toplam_val_kayip = 0.0
        dogru = 0
        toplam_cer_pay = 0
        toplam_cer_payda = 0
        hatalilar = []
        with torch.no_grad():
            for gorseller, hedefler_duz, uzunluklar, etiketler, dosyalar in val_yukleyici:
                gorseller = gorseller.to(cihaz)
                hedefler_duz_gpu = hedefler_duz.to(cihaz)
                logits = model(gorseller)
                log_olasilik = logits.log_softmax(2)
                T, B, _ = log_olasilik.shape
                girdi_uzunluklari = torch.full((B,), T, dtype=torch.long)
                kayip = kayip_fn(log_olasilik, hedefler_duz_gpu, girdi_uzunluklari, uzunluklar)
                toplam_val_kayip += kayip.item() * B

                tahminler = greedy_decode(logits, ters_index)
                for tahmin, gercek, dosya in zip(tahminler, etiketler, dosyalar):
                    if tahmin == gercek:
                        dogru += 1
                    else:
                        hatalilar.append((dosya, gercek, tahmin))
                    toplam_cer_pay += levenshtein(tahmin, gercek)
                    toplam_cer_payda += len(gercek)

        val_kayip = toplam_val_kayip / len(val_set)
        val_dogruluk = dogru / len(val_set)
        val_cer = toplam_cer_pay / max(1, toplam_cer_payda)

        scheduler.step(val_kayip)

        gecmis["epoch"].append(epoch)
        gecmis["train_kayip"].append(train_kayip)
        gecmis["val_kayip"].append(val_kayip)
        gecmis["train_kelime_dogruluk"].append(train_dogruluk)
        gecmis["val_kelime_dogruluk"].append(val_dogruluk)
        gecmis["val_karakter_hata_orani"].append(val_cer)

        guncel_lr = optimizer.param_groups[0]["lr"]
        print(f"[{epoch:3d}/{EPOCH_SAYISI}] egitim_kayip={train_kayip:.4f}  "
              f"egitim_dogruluk=%{train_dogruluk*100:5.2f}  "
              f"val_kayip={val_kayip:.4f}  val_dogruluk=%{val_dogruluk*100:5.2f}  "
              f"val_CER=%{val_cer*100:5.2f}  lr={guncel_lr:.2e}")

        if val_dogruluk > en_iyi_dogruluk:
            en_iyi_dogruluk = val_dogruluk
            en_iyi_epoch = epoch
            son_iyilesmeden_beri = 0
            model_kaydet(model, os.path.join(KLASOR, "model_en_iyi.pt"), sinif_sayisi)
            with open(os.path.join(KLASOR, "alfabe.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "alfabe": tum_karakterler,
                    "giris_yukseklik": oi.GIRIS_YUKSEKLIK,
                    "giris_genislik": oi.GIRIS_GENISLIK,
                    "cerceve": "pytorch",
                    "mimari": "CRNN+CTC",
                    "egitim_ornegi": len(train_set),
                    "dogrulama_ornegi": len(val_set),
                }, f, ensure_ascii=False, indent=2)
            with open(os.path.join(KLASOR, "val_hatalari.txt"), "w", encoding="utf-8") as f:
                f.write(f"Epoch {epoch} - val kelime dogrulugu: {val_dogruluk:.4f}\n")
                f.write(f"Yanlis tahmin sayisi: {len(hatalilar)} / {len(val_set)}\n\n")
                for dosya, gercek, tahmin in hatalilar:
                    f.write(f"{dosya}\tgercek={gercek}\ttahmin={tahmin}\n")
            print(f"   -> yeni en iyi model kaydedildi (val_kelime_dogruluk={val_dogruluk:.4f})")
        else:
            son_iyilesmeden_beri += 1
            if son_iyilesmeden_beri >= ERKEN_DURDURMA_SABRI:
                print(f"\n{ERKEN_DURDURMA_SABRI} epoch'tur val_dogruluk iyilesmedi "
                      f"(en iyi: epoch {en_iyi_epoch}, %{en_iyi_dogruluk*100:.2f}) - "
                      f"erken durduruluyor (kalan epoch bosa harcanmasin diye).")
                break

    # --- egitim sonu: grafik + rapor ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(gecmis["epoch"], gecmis["train_kayip"], label="egitim")
    plt.plot(gecmis["epoch"], gecmis["val_kayip"], label="dogrulama")
    plt.xlabel("epoch"); plt.ylabel("CTC kaybi"); plt.legend(); plt.title("Kayip")

    plt.subplot(1, 2, 2)
    plt.plot(gecmis["epoch"], gecmis["train_kelime_dogruluk"], label="egitim dogrulugu")
    plt.plot(gecmis["epoch"], gecmis["val_kelime_dogruluk"], label="dogrulama dogrulugu")
    plt.plot(gecmis["epoch"], [1 - c for c in gecmis["val_karakter_hata_orani"]], label="1 - val CER")
    plt.xlabel("epoch"); plt.legend(); plt.title("Dogruluk metrikleri")
    plt.tight_layout()
    plt.savefig(os.path.join(KLASOR, "egitim_gecmisi.png"), dpi=120)

    son_train_dogruluk = gecmis["train_kelime_dogruluk"][-1]
    son_val_cer = gecmis["val_karakter_hata_orani"][-1]

    karsilastirma_satiri = ""
    if onceki_dogruluk is not None:
        fark = (en_iyi_dogruluk - onceki_dogruluk) * 100
        if fark >= 0:
            karsilastirma_satiri = (f"Onceki modelden DAHA IYI: +{fark:.2f} puan "
                                     f"(onceki: %{onceki_dogruluk*100:.2f})\n")
        else:
            karsilastirma_satiri = (f"UYARI: Onceki modelden DAHA KOTU: {fark:.2f} puan "
                                     f"(onceki: %{onceki_dogruluk*100:.2f}) - eski model "
                                     f"model_yedekler/ klasorunde guvende, istersen oradan "
                                     f"geri kopyalayabilirsin.\n")

    with open(os.path.join(KLASOR, "egitim_raporu.txt"), "w", encoding="utf-8") as f:
        f.write(f"Toplam ornek (unsure=1 haric): {len(kayitlar)} "
                f"(egitim={len(train_set)}, dogrulama={len(val_set)})\n")
        f.write(f"Alfabe ({len(tum_karakterler)}): {''.join(tum_karakterler)}\n\n")
        f.write("=== DOGRULUK ORANLARI ===\n")
        f.write(f"Son epoch egitim (train) kelime dogrulugu : %{son_train_dogruluk*100:.2f}\n")
        f.write(f"En iyi dogrulama (val) kelime dogrulugu   : %{en_iyi_dogruluk*100:.2f}  "
                f"(epoch {en_iyi_epoch}/{EPOCH_SAYISI})\n")
        f.write(f"Son epoch dogrulama (val) karakter hata orani (CER): %{son_val_cer*100:.2f}  "
                f"(yani karakterlerin ~%{(1-son_val_cer)*100:.2f}'i dogru okunuyor)\n")
        if karsilastirma_satiri:
            f.write(f"\n{karsilastirma_satiri}")

    print("\n" + "=" * 50)
    print("EGITIM BITTI - DOGRULUK ORANLARI")
    print("=" * 50)
    print(f"Egitim (train) kelime dogrulugu (son epoch) : %{son_train_dogruluk*100:.2f}")
    print(f"Dogrulama (val) kelime dogrulugu (en iyi)    : %{en_iyi_dogruluk*100:.2f}  (epoch {en_iyi_epoch})")
    print(f"Dogrulama (val) karakter hata orani (CER)    : %{son_val_cer*100:.2f}")
    if karsilastirma_satiri:
        print(karsilastirma_satiri.strip())
    print("Ciktilar: model_en_iyi.pt, alfabe.json, egitim_gecmisi.png, egitim_raporu.txt, val_hatalari.txt")


if __name__ == "__main__":
    main()