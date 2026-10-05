"""
Praktikum 1 - Pengantar Logika Fuzzy dan Representasi Derajat Keanggotaan

Isi:
  Bagian A : Program modul (Crisp vs Fuzzy "Layanan Memuaskan") + analisis kondisi batas
  Bagian B : Praktikum mandiri (kurva Sigmoid x0 = 3.5, k = 2.0)
  Bagian C : Tugas - Sistem Prioritas Tiket Helpdesk TI (semesta 0..24 jam)

Library: hanya NumPy dan Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# BAGIAN A - PROGRAM MODUL
# ==========================================================

def crisp_memuaskan(rating, threshold=4.0):
    """Fungsi karakteristik crisp: 1 jika rating >= threshold, selain itu 0."""
    return np.where(np.asarray(rating, dtype=float) >= threshold, 1.0, 0.0)


def fuzzy_memuaskan(rating, a=2.5, b=4.5):
    """Fungsi keanggotaan linear naik: 0 jika x <= a, 1 jika x >= b, linear di antaranya."""
    derajat = (np.asarray(rating, dtype=float) - a) / (b - a)
    return np.clip(derajat, 0.0, 1.0)


def sigmoid(x, x0=3.5, k=2.0):
    """Praktikum mandiri: mu(x) = 1 / (1 + e^(-k(x - x0)))"""
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(x, dtype=float) - x0)))


def interpretasi(derajat):
    """Makna derajat keanggotaan (bukan persentase atau probabilitas)."""
    if derajat <= 0.0:
        return "Belum memuaskan"
    if derajat >= 1.0:
        return "Memuaskan penuh"
    return "Memuaskan sebagian"


# 401 titik -> jarak 0.01, sehingga nilai 3.9 dan 4.0 tepat ada pada grid
ratings = np.linspace(1.0, 5.0, 401)
y_crisp = crisp_memuaskan(ratings)
y_fuzzy = fuzzy_memuaskan(ratings)

plt.figure(figsize=(10, 5))
plt.step(ratings, y_crisp, label='Crisp (Threshold = 4.0)', color='#d9534f', linewidth=2.5, where='post')
plt.plot(ratings, y_fuzzy, label='Fuzzy (Linear Naik [2.5, 4.5])', color='#0275d8', linewidth=2.5)
plt.axvline(x=3.9, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=4.0, color='gray', linestyle='--', alpha=0.7)
plt.scatter([3.9, 4.0], crisp_memuaskan([3.9, 4.0]), color='#d9534f', zorder=5, s=60)
plt.scatter([3.9, 4.0], fuzzy_memuaskan([3.9, 4.0]), color='#0275d8', zorder=5, s=60)
plt.title('Perbandingan Logika Crisp vs Logika Fuzzy: Kategori "Layanan Memuaskan"',
          fontsize=13, fontweight='bold')
plt.xlabel('Rating Pengguna (Skala 1 - 5)')
plt.ylabel('Derajat Keanggotaan / Nilai Kebenaran')
plt.ylim(-0.05, 1.1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('visualisasi_crisp_vs_fuzzy.png', dpi=300)
plt.show()

print("=" * 70)
print("BAGIAN A - ANALISIS KONDISI BATAS (Layanan Memuaskan)")

print("=" * 70)
test_values = [3.8, 3.9, 3.99, 4.0, 4.01, 4.2]
print(f"{'Rating':<8} | {'Crisp':<6} | {'Fuzzy mu(x)':<12} | Interpretasi Fuzzy")
print("-" * 70)

for val in test_values:
    c_val = float(crisp_memuaskan(val))
    f_val = float(fuzzy_memuaskan(val))
    print(f"{val:<8.2f} | {c_val:<6.1f} | {f_val:<12.3f} | {interpretasi(f_val)}")

# ==========================================================
# BAGIAN B - PRAKTIKUM MANDIRI: SIGMOID
# ==========================================================

y_sig = sigmoid(ratings, x0=3.5, k=2.0)

plt.figure(figsize=(10, 5))
plt.step(ratings, y_crisp, label='Crisp (Threshold = 4.0)', color='#d9534f', linewidth=2.5, where='post')
plt.plot(ratings, y_fuzzy, label='Fuzzy Linear Naik [2.5, 4.5]', color='#0275d8', linewidth=2.5)
plt.plot(ratings, y_sig, label='Fuzzy Sigmoid (x0=3.5, k=2.0)', color='#2ca02c', linewidth=2.5)
plt.title('Crisp vs Linear Naik vs Sigmoid', fontsize=13, fontweight='bold')
plt.xlabel('Rating Pengguna (Skala 1 - 5)')
plt.ylabel('Derajat Keanggotaan / Nilai Kebenaran')
plt.ylim(-0.05, 1.1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('visualisasi_sigmoid.png', dpi=300)
plt.show()

print()

print("=" * 70)
print("BAGIAN B - SIGMOID (x0 = 3.5, k = 2.0)")

print("=" * 70)
print(f"{'Rating':<8} | {'Crisp':<6} | {'Linear':<8} | {'Sigmoid':<8}")
print("-" * 40)

for val in [1.0, 2.5, 3.0, 3.5, 3.9, 4.0, 4.5, 5.0]:
    print(f"{val:<8.2f} | {float(crisp_memuaskan(val)):<6.1f} | "
          f"{float(fuzzy_memuaskan(val)):<8.4f} | {float(sigmoid(val)):<8.4f}")

# ==========================================================
# BAGIAN C - TUGAS: PRIORITAS TIKET HELPDESK TI
# Variabel input : waktu tunggu tiket (jam), semesta 0 <= x <= 24
# Crisp : Kritis jika x >= 8
# Fuzzy : mu_Kritis(x) = 0 (x <= 4) ; (x - 4)/8 (4 < x < 12) ; 1 (x >= 12)
# ==========================================================

def crisp_kritis(x, threshold=8.0):
    return np.where(np.asarray(x, dtype=float) >= threshold, 1.0, 0.0)


def fuzzy_kritis(x, a=4.0, b=12.0):
    return np.clip((np.asarray(x, dtype=float) - a) / (b - a), 0.0, 1.0)


def interpretasi_kritis(derajat):
    if derajat <= 0.0:
        return "Belum kritis"
    if derajat >= 1.0:
        return "Kritis penuh"
    return "Kritis sebagian"


data_uji = [2, 4, 6, 7.9, 8.0, 8.1, 10, 12, 16, 24]

print()

print("=" * 78)
print("BAGIAN C - TUGAS: TIKET HELPDESK TI (Kritis / Butuh Eskalasi Cepat)")

print("=" * 78)
print("Rumus fuzzy: mu(x) = 0 (x <= 4) ; (x-4)/(12-4) = (x-4)/8 (4 < x < 12) ; 1 (x >= 12)")
print()
print("| No | Waktu (jam) | Substitusi           | Crisp | Fuzzy mu(x) | Interpretasi    |")
print("|---:|------------:|----------------------|------:|------------:|-----------------|")

for i, t in enumerate(data_uji, 1):
    c = float(crisp_kritis(t))
    f = float(fuzzy_kritis(t))
    if t <= 4:
        sub = "x <= 4 -> 0"
    elif t < 12:
        sub = f"({t}-4)/8 = {(t - 4):.1f}/8"
    else:
        sub = "x >= 12 -> 1"
    print(f"| {i:>2} | {t:>11} | {sub:<20} | {c:>5.1f} | {f:>11.4f} | {interpretasi_kritis(f):<15} |")

x_t = np.linspace(0, 24, 241)   # jarak 0.1 jam, sehingga 7.9 / 8.0 / 8.1 tepat pada grid

plt.figure(figsize=(10, 5))
plt.step(x_t, crisp_kritis(x_t), where='post', color='#d9534f', linewidth=2.5, label='Crisp (>= 8 jam)')
plt.plot(x_t, fuzzy_kritis(x_t), color='#0275d8', linewidth=2.5, label='Fuzzy (transisi 4 - 12 jam)')
plt.scatter(data_uji, crisp_kritis(data_uji), color='#d9534f', zorder=5, s=45)
plt.scatter(data_uji, fuzzy_kritis(data_uji), color='#0275d8', zorder=5, s=45)
plt.axvline(7.9, color='gray', linestyle='--', alpha=0.6)
plt.axvline(8.0, color='gray', linestyle='--', alpha=0.6)
plt.title('Tiket Helpdesk TI: Crisp vs Fuzzy "Kritis / Butuh Eskalasi Cepat"',
          fontsize=13, fontweight='bold')
plt.xlabel('Waktu Tunggu Tiket (jam)')
plt.ylabel('Derajat Keanggotaan pada Himpunan Kritis')
plt.xlim(0, 24)
plt.ylim(-0.05, 1.1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='center right')
plt.tight_layout()
plt.savefig('visualisasi_tiket_helpdesk.png', dpi=300)
plt.show()

print("\nGrafik tersimpan: visualisasi_crisp_vs_fuzzy.png, visualisasi_sigmoid.png, visualisasi_tiket_helpdesk.png")
