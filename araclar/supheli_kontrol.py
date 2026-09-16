# -*- coding: utf-8 -*-
"""
SUPHELI ETIKETLERI ELLE KONTROL ARAYUZU
==========================================
labels_lower.csv icindeki unsure=1 (etiketleyenin emin olmadigi) satirlari
tek tek listeler: gorsel + mevcut etiket + (varsa) model_en_iyi.pt'nin
tahmini yan yana gosterilir. Her satirda:
  - Metni duzeltip "Kaydet" -> label guncellenir, unsure=0 yapilir; ayni
    goruntunun etiketli/ klasorundeki kopyasi da (icerik hash'i ile bulunup)
    YENI etiket adina gore otomatik yeniden adlandirilir - CSV ile etiketli/
    her zaman senkron kalir.
  - "Sil" -> satir CSV'den silinir, gorsel KALICI SILINMEZ, guvenlik icin
    cptchYeni/veriler/silinenler/ klasorune tasinir (geri alinabilir); ayni
    sekilde etiketli/ klasorundeki kopyasi da oraya tasinir.

Calistirma:
    python supheli_kontrol.py
Sonra tarayicida ac: http://127.0.0.1:5050
"""
import os
import csv
import json
import shutil
import hashlib

import numpy as np
import torch
from flask import Flask, request, redirect, send_file, abort

import onisleme as oi
from captcha_model import model_yukle, cihaz_sec

KLASOR = os.path.dirname(os.path.abspath(__file__))
VERI_DIZIN = os.path.join(KLASOR, "cptchYeni", "veriler")
LABELS_CSV = os.path.join(VERI_DIZIN, "labels_lower.csv")
SILINEN_DIZIN = os.path.join(VERI_DIZIN, "silinenler")
ETIKETLI_DIZIN = os.path.join(VERI_DIZIN, "etiketli")
MODEL_YOLU = os.path.join(KLASOR, "model_en_iyi.pt")
ALFABE_YOLU = os.path.join(KLASOR, "alfabe.json")

FIELDNAMES = ["filename", "label", "unsure"]

app = Flask(__name__)

# ============================ MODEL (tahmin icin) ============================
cihaz = None
model = None
ters_index = {}


def modeli_yukle_varsa():
    global model, cihaz
    if not (os.path.exists(MODEL_YOLU) and os.path.exists(ALFABE_YOLU)):
        print("UYARI: model_en_iyi.pt / alfabe.json bulunamadi - tahminler '(model yok)' gosterilecek.")
        return
    with open(ALFABE_YOLU, encoding="utf-8") as f:
        alfabe = json.load(f)["alfabe"]
    ters_index.clear()
    ters_index.update({i + 1: c for i, c in enumerate(alfabe)})
    cihaz = cihaz_sec()
    model = model_yukle(MODEL_YOLU, cihaz)


def greedy_decode_tek(logits):
    """logits: (T,1,sinif_sayisi) -> tek ornek icin metin."""
    tahmin = logits.argmax(-1).squeeze(1).cpu().numpy()  # (T,)
    metin = []
    onceki = -1
    for k in tahmin:
        if k != 0 and k != onceki:
            metin.append(ters_index.get(int(k), "?"))
        onceki = k
    return "".join(metin)


def tahmin_et(dosya_yolu):
    if model is None:
        return "(model yok)"
    gri = oi.gorsel_oku(dosya_yolu)
    if gri is None:
        return "(okunamadi)"
    gri = oi.on_isle(gri)
    x = torch.from_numpy((gri.astype(np.float32) / 255.0)[None, None, :, :]).to(cihaz)
    with torch.no_grad():
        logits = model(x)
    return greedy_decode_tek(logits)


# ============================ ETIKETLI/ SENKRONU ============================
# etiketli/ klasorundeki dosyalar "veriler" kokundeki gorsellerin ayni
# icerikli (byte-byte) kopyalari, sadece dosya adi etiket metni. CSV'deki bir
# satirin etiketi degisince/silininde bu kopyayi da bulup senkron tutmak icin
# icerik hash'i uzerinden eslestiriyoruz (dosya adindan degil).
etiketli_hash_index = {}  # md5 -> etiketli/ icindeki dosya adi


def dosya_hash(yol):
    h = hashlib.md5()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(65536), b""):
            h.update(parca)
    return h.hexdigest()


def etiketli_index_olustur():
    etiketli_hash_index.clear()
    if not os.path.isdir(ETIKETLI_DIZIN):
        return
    for ad in os.listdir(ETIKETLI_DIZIN):
        yol = os.path.join(ETIKETLI_DIZIN, ad)
        if os.path.isfile(yol):
            etiketli_hash_index[dosya_hash(yol)] = ad
    print(f"etiketli/ indeksi olusturuldu: {len(etiketli_hash_index)} dosya")


def etiketli_esini_bul(kaynak_yol):
    """kaynak_yol ile ayni icerige sahip etiketli/ dosyasinin adini dondurur (yoksa None)."""
    return etiketli_hash_index.get(dosya_hash(kaynak_yol))


def benzersiz_etiketli_adi(etiket, haric=None):
    """etiketli/ icinde 'etiket.png' zaten varsa etiket_2.png, etiket_3.png... dener."""
    mevcutlar = set(os.listdir(ETIKETLI_DIZIN)) if os.path.isdir(ETIKETLI_DIZIN) else set()
    if haric:
        mevcutlar.discard(haric)
    ad = f"{etiket}.png"
    sayac = 2
    while ad in mevcutlar:
        ad = f"{etiket}_{sayac}.png"
        sayac += 1
    return ad


# ============================ CSV OKU/YAZ ============================
def csv_oku():
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def csv_yaz(satirlar):
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        yazici = csv.DictWriter(f, fieldnames=FIELDNAMES)
        yazici.writeheader()
        yazici.writerows(satirlar)


# ============================ ROTALAR ============================
@app.route("/")
def anasayfa():
    satirlar = csv_oku()
    supheliler = [s for s in satirlar if s.get("unsure") == "1"]

    satir_html = []
    for s in supheliler:
        dosya = s["filename"]
        etiket = s["label"]
        yol = os.path.join(VERI_DIZIN, dosya)
        tahmin = tahmin_et(yol) if os.path.exists(yol) else "(dosya yok)"
        satir_html.append(f"""
        <div class="kart">
          <img src="/gorsel/{dosya}" alt="{dosya}">
          <div class="bilgi">
            <div><b>Dosya:</b> {dosya}</div>
            <div><b>Model tahmini:</b> <span class="tahmin">{tahmin}</span></div>
            <form method="post" action="/kaydet" class="satir">
              <input type="hidden" name="dosya" value="{dosya}">
              <input type="text" name="yeni_etiket" value="{etiket}">
              <button type="submit">Kaydet</button>
            </form>
            <form method="post" action="/sil" onsubmit="return confirm('Bu ornek silinsin mi?');">
              <input type="hidden" name="dosya" value="{dosya}">
              <button type="submit" class="sil">Sil</button>
            </form>
          </div>
        </div>
        """)

    return f"""
    <html><head><meta charset="utf-8">
    <title>Supheli Etiketler</title>
    <style>
      body {{ font-family: sans-serif; background:#111; color:#eee; margin:0; }}
      h2 {{ padding:16px 16px 0 16px; }}
      .grid {{ display:flex; flex-wrap:wrap; gap:16px; padding:16px; }}
      .kart {{ background:#1c1c1c; border:1px solid #333; border-radius:8px; padding:10px; width:300px; }}
      .kart img {{ width:100%; border-radius:4px; background:#fff; }}
      .satir {{ display:flex; gap:6px; margin-top:6px; }}
      input[type=text] {{ flex:1; font-size:14px; }}
      button {{ cursor:pointer; }}
      .sil {{ background:#822; color:#fff; border:none; padding:4px 8px; border-radius:4px; margin-top:6px; width:100%; }}
      .tahmin {{ color:#6cf; font-weight:bold; }}
    </style></head>
    <body>
      <h2>Supheli (unsure=1) etiketler: {len(supheliler)} adet</h2>
      <div class="grid">{''.join(satir_html) if satir_html else '<p style="padding:16px">Supheli etiket kalmadi.</p>'}</div>
    </body></html>
    """


@app.route("/gorsel/<path:dosya>")
def gorsel(dosya):
    if "/" in dosya or "\\" in dosya or ".." in dosya:
        abort(400)
    yol = os.path.join(VERI_DIZIN, dosya)
    if not os.path.exists(yol):
        abort(404)
    return send_file(yol)


@app.route("/kaydet", methods=["POST"])
def kaydet():
    dosya = request.form["dosya"]
    yeni_etiket = request.form["yeni_etiket"].strip()
    satirlar = csv_oku()
    for s in satirlar:
        if s["filename"] == dosya:
            s["label"] = yeni_etiket
            s["unsure"] = "0"
    csv_yaz(satirlar)

    # etiketli/ klasorundeki karsilik gelen kopyayi da yeni etikete gore
    # yeniden adlandir (CSV ile senkron kalsin)
    kaynak_yol = os.path.join(VERI_DIZIN, dosya)
    if os.path.exists(kaynak_yol):
        eski_ad = etiketli_esini_bul(kaynak_yol)
        if eski_ad and eski_ad != f"{yeni_etiket}.png":
            yeni_ad = benzersiz_etiketli_adi(yeni_etiket, haric=eski_ad)
            os.rename(os.path.join(ETIKETLI_DIZIN, eski_ad), os.path.join(ETIKETLI_DIZIN, yeni_ad))
            etiketli_hash_index[dosya_hash(kaynak_yol)] = yeni_ad

    return redirect("/")


@app.route("/sil", methods=["POST"])
def sil():
    dosya = request.form["dosya"]
    satirlar = csv_oku()
    satirlar = [s for s in satirlar if s["filename"] != dosya]
    csv_yaz(satirlar)
    # kalici silmek yerine 'silinenler' klasorune tasi (geri alinabilir)
    os.makedirs(SILINEN_DIZIN, exist_ok=True)
    kaynak = os.path.join(VERI_DIZIN, dosya)
    if os.path.exists(kaynak):
        eski_ad = etiketli_esini_bul(kaynak)
        if eski_ad:
            eski_hash = dosya_hash(kaynak)
            shutil.move(os.path.join(ETIKETLI_DIZIN, eski_ad), os.path.join(SILINEN_DIZIN, eski_ad))
            etiketli_hash_index.pop(eski_hash, None)
        shutil.move(kaynak, os.path.join(SILINEN_DIZIN, dosya))
    return redirect("/")


if __name__ == "__main__":
    modeli_yukle_varsa()
    etiketli_index_olustur()
    print("Acmak icin tarayicida git: http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)
