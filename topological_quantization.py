import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

# ============================================================
# 1. PARAMETRELER
# ============================================================
N = 128
L = 10.0
dx = L / N
dt = 0.001

x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# ============================================================
# 2. BAŞLANGIÇ KOŞULU (w = 0.73)
# ============================================================
w_init = 0.73
a, b = 2.5, 0.8
rho = np.exp(-(X**2 / a**2 + Y**2 / b**2))
chi = w_init * np.arctan2(Y, X)  # Başlangıçta w=0.73 olan faz

phi1 = rho * np.cos(chi)
phi2 = rho * np.sin(chi)

# ============================================================
# 3. FARKLI İTERASYON ADIMLARINDA w'Yİ TAKİP ET
# ============================================================
# İzlenecek zaman adımları (sn cinsinden)
check_times = [1000, 5000, 10000, 50000]
w_at_checkpoints = []

print("="*70)
print(f"w = {w_init} BAŞLANGIÇ DEĞERİ İÇİN UZUN GEVŞEME TESTİ")
print("="*70)
print(f"Adım sayısı (zaman) -> w değeri")

# Her zaman diliminde simülasyonu ilerlet ve w'yi ölç
for i, n_steps in enumerate(check_times):
    # Eğer zaten ilerlediysek kaldığımız yerden devam et
    start_step = 0 if i == 0 else check_times[i-1]
    for step in range(start_step, n_steps):
        # Allen-Cahn adımı
        lap1 = (np.roll(phi1, 1, axis=0) + np.roll(phi1, -1, axis=0) +
                np.roll(phi1, 1, axis=1) + np.roll(phi1, -1, axis=1) - 4*phi1) / (dx**2)
        lap2 = (np.roll(phi2, 1, axis=0) + np.roll(phi2, -1, axis=0) +
                np.roll(phi2, 1, axis=1) + np.roll(phi2, -1, axis=1) - 4*phi2) / (dx**2)
        
        norm = phi1**2 + phi2**2
        V1 = (norm - 1) * phi1
        V2 = (norm - 1) * phi2
        
        phi1 = phi1 + dt * (lap1 - V1)
        phi2 = phi2 + dt * (lap2 - V2)
    
    # Bu adımda w'yi hesapla
    n_theta, n_phi = 60, 60
    theta_grid = np.linspace(0, np.pi, n_theta)
    phi_grid = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    
    X_target = (L/2) * np.sin(THETA) * np.cos(PHI)
    Y_target = (L/2) * np.sin(THETA) * np.sin(PHI)
    
    coords = np.array([(Y_target / (L/2) + 1) * (N-1)/2, 
                       (X_target / (L/2) + 1) * (N-1)/2])
    
    phi1_sph = map_coordinates(phi1, coords, order=1, mode='nearest')
    phi2_sph = map_coordinates(phi2, coords, order=1, mode='nearest')
    
    dphi = 2.0 * np.pi / n_phi
    dphi1 = np.gradient(phi1_sph, axis=1) / dphi
    dphi2 = np.gradient(phi2_sph, axis=1) / dphi
    
    norm_sph = phi1_sph**2 + phi2_sph**2 + 1e-8
    A_phi = (phi1_sph * dphi2 - phi2_sph * dphi1) / norm_sph
    
    mean_A_phi = np.mean(A_phi, axis=0)
    w_current = np.sum(mean_A_phi) * dphi / (2 * np.pi)
    
    w_at_checkpoints.append(w_current)
    print(f"  {n_steps:6d} adım -> w = {w_current:.6f}")

# ============================================================
# 4. SONUÇ GRAFİĞİ
# ============================================================
plt.figure(figsize=(10, 6))
plt.plot(check_times, w_at_checkpoints, 'bo-', linewidth=2, markersize=8)
plt.axhline(0.0, color='blue', linestyle='--', alpha=0.5, label='w = 0 (Boson / SO(3))')
plt.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='w = 0.5 (Fermiyon / SU(2))')
plt.axhline(1.0, color='orange', linestyle='--', alpha=0.5, label='w = 1.0 (Boson)')
plt.axhline(w_init, color='red', linestyle=':', alpha=0.3, label=f'Başlangıç w = {w_init}')

plt.xlabel('İterasyon Sayısı (Adım)')
plt.ylabel('Hesaplanan w Değeri')
plt.title(f'w = {w_init} İçin Uzun Gevşeme Testi')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("FİZİKSEL YORUM")
print("="*70)
if abs(w_at_checkpoints[-1]) < 0.01:
    print(">> Sistem tamamen SO(3) (w = 0) sınıfına kilitlendi.")
elif abs(w_at_checkpoints[-1] - 0.5) < 0.01:
    print(">> Sistem SU(2) (w = 0.5) sınıfına kilitlendi!")
elif abs(w_at_checkpoints[-1] - 1.0) < 0.01:
    print(">> Sistem w = 1.0 sınıfına kilitlendi.")
else:
    print(f">> Sistem {w_at_checkpoints[-1]:.4f} civarında kaldı. (Ara topolojik durum)")