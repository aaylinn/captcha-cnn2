# -*- coding: utf-8 -*-
"""
LUCAMODEL - TEK GORSEL ICIN TAHMIN
=====================================
Bu klasordeki egitilmis modeli kullanarak tek bir CAPTCHA gorseli icin
metin tahmini yapar. Bagimsiz calisir - sadece bu klasordeki dosyalara
(lucamodel.pt, alfabe.json, captcha_model.py, onisleme.py) ihtiyac duyar.

Kurulum:
    pip install -r requirements.txt

Kullanim:
    python tahmin_et.py yol/to/captcha.png
"""
import sys
import os
import json

import numpy as np
import torch

import onisleme as oi
from captcha_model import model_yukle, cihaz_sec

KLASOR = os.path.dirname(os.path.abspath(__file__))
MODEL_YOLU = os.path.join(KLASOR, "lucamodel.pt")
ALFABE_YOLU = os.path.join(KLASOR, "alfabe.json")


def greedy_decode(logits, ters_index):
    """logits: (T,1,sinif_sayisi) -> tek ornek icin metin.
    Greedy: her adimda en olasi sinif, ardindan ardisik tekrarlar ve
    'bos' (CTC blank, 0) sinifi silinir."""
    tahmin = logits.argmax(-1).squeeze(1).cpu().numpy()  # (T,)
    metin = []
    onceki = -1
    for k in tahmin:
        if k != 0 and k != onceki:
            metin.append(ters_index.get(int(k), "?"))
        onceki = k
    return "".join(metin)


def tahmin_et(gorsel_yolu, model=None, cihaz=None, ters_index=None):
    """Tek bir gorsel dosyasi icin metin tahmini dondurur."""
    if model is None:
        cihaz = cihaz_sec()
        model = model_yukle(MODEL_YOLU, cihaz)
        with open(ALFABE_YOLU, encoding="utf-8") as f:
            alfabe = json.load(f)["alfabe"]
        ters_index = {i + 1: c for i, c in enumerate(alfabe)}

    gri = oi.gorsel_oku(gorsel_yolu)
    if gri is None:
        raise ValueError(f"Gorsel okunamadi: {gorsel_yolu}")
    gri = oi.on_isle(gri)
    x = torch.from_numpy((gri.astype(np.float32) / 255.0)[None, None, :, :]).to(cihaz)
    with torch.no_grad():
        logits = model(x)
    return greedy_decode(logits, ters_index)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Kullanim: python tahmin_et.py yol/to/captcha.png")
        sys.exit(1)

    gorsel_yolu = sys.argv[1]
    if not os.path.exists(gorsel_yolu):
        print(f"HATA: dosya bulunamadi: {gorsel_yolu}")
        sys.exit(1)

    tahmin = tahmin_et(gorsel_yolu)
    print(f"Tahmin: {tahmin}")
