# -*- coding: utf-8 -*-
"""
YENI VERI - MODEL YARDIMLI TEK-TEK ETIKETLEME ARAYUZU
========================================================
cptchYeni/veriler/yeniVeriler/ klasorundeki HENUZ ETIKETLENMEMIS gorselleri
tek tek gosterir. Her gorsel icin model_en_iyi.pt'nin tahmini bir metin
kutusuna ONCEDEN yazilir:
  - Tahmin dogruysa -> direkt "Kaydet"
  - Tahmin yanlissa  -> kutuyu duzelt, sonra "Kaydet"
  - Bu ornegi hic kullanmak istemiyorsan -> "Cikar"

"Kaydet": gorsel cptchYeni/veriler/yeniVeriler/ 'dan cptchYeni/veriler/ (kok)
klasorune TASINIR, labels_lower.csv'ye unsure=0 olarak eklenir, etiketli/
klasorune de etiket adiyla kopyalanir - captcha_egit.py bir sonraki
calistirmada bu ornegi otomatik egitime dahil eder.

"Cikar": gorsel KALICI SILINMEZ, cptchYeni/veriler/yeniVeriler_reddedilenler/
klasorune tasinir (geri alinabilir), veri setine hic girmez.

OTOMATIK ON-ONAY (sunucu her baslarken bir kere calisir):
Model her tahmin icin bir de GUVEN skoru uretir (secilen karakterlerin
ortalama softmax olasiligi, 0-1 arasi). GUVEN_ESIGI'nin USTUNDEKI tahminler
hic ekrana gelmeden otomatik olarak dogru kabul edilip kaydedilir (Kaydet
butonuyla ayni islem: tasi + csv'ye ekle + etiketliye kopyala). Sadece
ESIGIN ALTINDA kalanlar bu arayuzde elle kontrol icin gosterilir.

Kuyruk (ilerleme durumu) ayri bir dosyada tutulmuyor - yeniVeriler/
klasorundeki kalan dosyalar dogrudan kuyruktur. Bu sayede sunucuyu
yeniden baslatinca kaldigin yerden devam eder (islenen dosyalar klasorden
cikmis olur).

Calistirma:
    python yeni_veri_etiketle.py
Sonra tarayicida ac: http://127.0.0.1:5051
"""
import os
import re
import csv
import json
import shutil

import numpy as np
import torch
from flask import Flask, request, redirect, send_file, abort

import onisleme as oi
from captcha_model import model_yukle, cihaz_sec

KLASOR = os.path.dirname(os.path.abspath(__file__))
VERI_DIZIN = os.path.join(KLASOR, "cptchYeni", "veriler")
YENI_VERI_DIZIN = os.path.join(VERI_DIZIN, "yeniVeriler")
REDDEDILEN_DIZIN = os.path.join(VERI_DIZIN, "yeniVeriler_reddedilenler")
ETIKETLI_DIZIN = os.path.join(VERI_DIZIN, "etiketli")
LABELS_CSV = os.path.join(VERI_DIZIN, "labels_lower.csv")
MODEL_YOLU = os.path.join(KLASOR, "model_en_iyi.pt")
ALFABE_YOLU = os.path.join(KLASOR, "alfabe.json")

FIELDNAMES = ["filename", "label", "unsure"]
DOSYA_DESENI = re.compile(r"^screenshot_\d+\.png$")

# Bu esigin USTUNDEKI tahminler otomatik kaydedilir, ALTINDAKILER elle
# kontrol icin gosterilir. Model su an ~%76 tam-kelime dogrulugunda ama
# "guven skoru yuksek" olan tahminler tipik olarak ortalamadan cok daha
# guvenilirdir - 0.90 ile baslamak makul; val_hatalari.txt'ye gore cok
# fazla hatali otomatik kaydediliyorsa yukselt (orn 0.95), az sayida
# guvenilir tahmin de elle kontrole dusuyorsa dusur.
GUVEN_ESIGI = 0.90

app = Flask(__name__)

# ============================ MODEL (tahmin icin) ============================
cihaz = None
model = None
ters_index = {}


def modeli_yukle_varsa():
    global model, cihaz
    if not (os.path.exists(MODEL_YOLU) and os.path.exists(ALFABE_YOLU)):
        print("UYARI: model_en_iyi.pt / alfabe.json bulunamadi - tahmin kutusu bos gelecek.")
        return
    with open(ALFABE_YOLU, encoding="utf-8") as f:
        alfabe = json.load(f)["alfabe"]
    ters_index.clear()
    ters_index.update({i + 1: c for i, c in enumerate(alfabe)})
    cihaz = cihaz_sec()
    model = model_yukle(MODEL_YOLU, cihaz)


def greedy_decode_guvenli(logits):
    """logits: (T,1,sinif_sayisi) -> (metin, guven).
    guven = secilen (bos/tekrar disi) karakterlerin ortalama softmax
    olasiligi - modelin kendi tahminine ne kadar 'emin' oldugunun basit
    ama standart bir olcusu (CTC greedy decode icin yaygin kullanilir)."""
    olasiliklar = logits.softmax(-1).squeeze(1)          # (T,sinif_sayisi)
    degerler, indeksler = olasiliklar.max(-1)             # (T,), (T,)
    degerler = degerler.cpu().numpy()
    indeksler = indeksler.cpu().numpy()

    metin = []
    guvenler = []
    onceki = -1
    for k, p in zip(indeksler, degerler):
        if k != 0 and k != onceki:
            metin.append(ters_index.get(int(k), "?"))
            guvenler.append(p)
        onceki = k
    guven = float(np.mean(guvenler)) if guvenler else 0.0
    return "".join(metin), guven


def tahmin_ve_guven(dosya_yolu):
    """Basariliysa (metin, guven) dondurur, model/gorsel yoksa ('', 0.0)."""
    if model is None:
        return "", 0.0
    gri = oi.gorsel_oku(dosya_yolu)
    if gri is None:
        return "", 0.0
    gri = oi.on_isle(gri)
    x = torch.from_numpy((gri.astype(np.float32) / 255.0)[None, None, :, :]).to(cihaz)
    with torch.no_grad():
        logits = model(x)
    return greedy_decode_guvenli(logits)


# ============================ KUYRUK ============================
def kuyruk():
    """yeniVeriler/ klasorundeki islenmemis dosyalari numaraya gore sirali dondurur."""
    if not os.path.isdir(YENI_VERI_DIZIN):
        return []
    dosyalar = [f for f in os.listdir(YENI_VERI_DIZIN) if DOSYA_DESENI.match(f)]
    dosyalar.sort(key=lambda f: int(re.search(r"\d+", f).group()))
    return dosyalar


TOPLAM_BASLANGIC = None  # sunucu baslarken bir kere hesaplanir, ilerleme cubugu icin


def benzersiz_etiketli_adi(etiket):
    mevcutlar = set(os.listdir(ETIKETLI_DIZIN)) if os.path.isdir(ETIKETLI_DIZIN) else set()
    ad = f"{etiket}.png"
    sayac = 2
    while ad in mevcutlar:
        ad = f"{etiket}_{sayac}.png"
        sayac += 1
    return ad


def dosyayi_kaydet(dosya, etiket):
    """Tek bir ornegi kesin etiketli veri setine ekler (elle Kaydet butonu VE
    otomatik on-onay gecisi bunu kullanir): yeniVeriler/'dan veriler/ kokune
    tasi, labels_lower.csv'ye ekle, etiketli/'ye kopyala."""
    kaynak = os.path.join(YENI_VERI_DIZIN, dosya)
    if not os.path.exists(kaynak) or not etiket:
        return False

    hedef = os.path.join(VERI_DIZIN, dosya)
    shutil.move(kaynak, hedef)

    with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
        yazici = csv.DictWriter(f, fieldnames=FIELDNAMES)
        yazici.writerow({"filename": dosya, "label": etiket, "unsure": "0"})

    os.makedirs(ETIKETLI_DIZIN, exist_ok=True)
    etiketli_ad = benzersiz_etiketli_adi(etiket)
    shutil.copy(hedef, os.path.join(ETIKETLI_DIZIN, etiketli_ad))
    return True


def otomatik_onayla_gecisi():
    """Kuyruktaki her gorsel icin tahmin+guven hesaplar; GUVEN_ESIGI USTUNDEKI
    tahminleri hic ekrana getirmeden otomatik kaydeder. Sadece sunucu
    baslarken bir kere calisir (ilerleme konsola yazilir)."""
    if model is None:
        print("Model yuklenemedigi icin otomatik on-onay atlandi.")
        return 0, 0

    baslangic_kuyruk = kuyruk()
    print(f"Otomatik on-onay basliyor: {len(baslangic_kuyruk)} gorsel, "
          f"guven esigi=%{GUVEN_ESIGI*100:.0f}")

    otomatik = 0
    elle_kontrol = 0
    for i, dosya in enumerate(baslangic_kuyruk, start=1):
        yol = os.path.join(YENI_VERI_DIZIN, dosya)
        if not os.path.exists(yol):
            continue
        tahmin, guven = tahmin_ve_guven(yol)
        if guven >= GUVEN_ESIGI and tahmin:
            dosyayi_kaydet(dosya, tahmin)
            otomatik += 1
        else:
            elle_kontrol += 1

        if i % 200 == 0 or i == len(baslangic_kuyruk):
            print(f"  ... {i}/{len(baslangic_kuyruk)} islendi "
                  f"(otomatik={otomatik}, elle_kontrol={elle_kontrol})")

    print(f"Otomatik on-onay bitti: {otomatik} otomatik kaydedildi, "
          f"{elle_kontrol} elle kontrol icin birakildi.")
    return otomatik, elle_kontrol


# ============================ ROTALAR ============================
@app.route("/")
def anasayfa():
    kalanlar = kuyruk()
    kalan = len(kalanlar)
    islenen = (TOPLAM_BASLANGIC - kalan) if TOPLAM_BASLANGIC else 0
    yuzde = (islenen / TOPLAM_BASLANGIC * 100) if TOPLAM_BASLANGIC else 0

    if not kalanlar:
        return f"""
        <html><head><meta charset="utf-8"><title>Yeni Veri Etiketleme</title>
        <style>body{{font-family:sans-serif;background:#111;color:#eee;
        display:flex;align-items:center;justify-content:center;height:100vh;margin:0;}}</style>
        </head><body><h2>🎉 yeniVeriler/ klasorunde islenecek dosya kalmadi.</h2></body></html>
        """

    dosya = kalanlar[0]
    yol = os.path.join(YENI_VERI_DIZIN, dosya)
    tahmin, guven = tahmin_ve_guven(yol)

    return f"""
    <html><head><meta charset="utf-8">
    <title>Yeni Veri Etiketleme</title>
    <style>
      body {{ font-family: sans-serif; background:#111; color:#eee; margin:0;
             display:flex; align-items:center; justify-content:center; min-height:100vh; }}
      .kart {{ background:#1c1c1c; border:1px solid #333; border-radius:12px; padding:24px;
              width:360px; text-align:center; }}
      .kart img {{ width:100%; border-radius:8px; background:#fff; margin-bottom:16px; }}
      .ilerleme {{ color:#888; font-size:13px; margin-bottom:12px; }}
      .bar {{ background:#333; border-radius:6px; height:6px; margin-bottom:16px; overflow:hidden; }}
      .bar-ic {{ background:#4c8; height:100%; width:{yuzde:.2f}%; }}
      input[type=text] {{ width:100%; font-size:22px; text-align:center; padding:10px;
                          border-radius:6px; border:1px solid #444; background:#000; color:#0f0;
                          box-sizing:border-box; margin-bottom:12px; letter-spacing:1px; }}
      .butonlar {{ display:flex; gap:8px; }}
      button {{ flex:1; padding:12px; font-size:15px; border:none; border-radius:6px; cursor:pointer; }}
      .kaydet {{ background:#2a7; color:#fff; }}
      .cikar {{ background:#822; color:#fff; }}
      .dosya-adi {{ color:#666; font-size:11px; margin-top:10px; }}
    </style></head>
    <body>
      <div class="kart">
        <div class="ilerleme">{islenen} / {TOPLAM_BASLANGIC} islendi ({kalan} kaldi) — bu ornek guven=%{guven*100:.1f} (esik %{GUVEN_ESIGI*100:.0f} altinda, elle kontrol)</div>
        <div class="bar"><div class="bar-ic"></div></div>
        <img src="/gorsel/{dosya}" alt="{dosya}">
        <form method="post" action="/kaydet" id="form-kaydet">
          <input type="hidden" name="dosya" value="{dosya}">
          <input type="text" name="etiket" value="{tahmin}" autofocus
                 onfocus="this.select()" autocomplete="off">
          <div class="butonlar">
            <button type="submit" class="kaydet">Kaydet (Enter)</button>
          </div>
        </form>
        <form method="post" action="/cikar" onsubmit="return confirm('Bu ornek disari cikarilsin mi?');">
          <input type="hidden" name="dosya" value="{dosya}">
          <div class="butonlar" style="margin-top:8px;">
            <button type="submit" class="cikar">Çıkar</button>
          </div>
        </form>
        <div class="dosya-adi">{dosya}</div>
      </div>
      <script>
        // Esc tusuyla hizlica "Cikar" yapabilmek icin
        document.addEventListener('keydown', function(e) {{
          if (e.key === 'Escape') {{
            document.querySelector('form[action="/cikar"]').requestSubmit();
          }}
        }});
      </script>
    </body></html>
    """


@app.route("/gorsel/<path:dosya>")
def gorsel(dosya):
    if "/" in dosya or "\\" in dosya or ".." in dosya:
        abort(400)
    yol = os.path.join(YENI_VERI_DIZIN, dosya)
    if not os.path.exists(yol):
        abort(404)
    return send_file(yol)


@app.route("/kaydet", methods=["POST"])
def kaydet():
    dosya = request.form["dosya"]
    etiket = request.form["etiket"].strip()
    dosyayi_kaydet(dosya, etiket)
    return redirect("/")


@app.route("/cikar", methods=["POST"])
def cikar():
    dosya = request.form["dosya"]
    kaynak = os.path.join(YENI_VERI_DIZIN, dosya)
    if os.path.exists(kaynak):
        os.makedirs(REDDEDILEN_DIZIN, exist_ok=True)
        shutil.move(kaynak, os.path.join(REDDEDILEN_DIZIN, dosya))
    return redirect("/")


if __name__ == "__main__":
    modeli_yukle_varsa()
    TOPLAM_BASLANGIC = len(kuyruk())
    print(f"Islenecek gorsel sayisi: {TOPLAM_BASLANGIC}")
    otomatik_onayla_gecisi()
    print("Acmak icin tarayicida git: http://127.0.0.1:5051")
    app.run(host="127.0.0.1", port=5051, debug=False)
