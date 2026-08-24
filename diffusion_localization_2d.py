import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import maximum_filter, label
from scipy.optimize import curve_fit
from scipy.stats import lognorm
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PARAMETRELER
# ==========================================
D0, kc, delta_k = 1.0, 0.8, 0.3
gamma0, lc, hbar_over_m = 1.0, 1.0, 1.0
N, L = 128, 20.0
dx = L / N
dt = 2.5e-5
total_steps = 2000

kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2

dealiasing_mask = (np.abs(KX) < (2.0/3.0) * (np.pi/dx)) & (np.abs(KY) < (2.0/3.0) * (np.pi/dx))

x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# ==========================================
# 2. SPEKTRAL YÖNTEM FONKSİYONLARI
# ==========================================
def apply_filter(f):
    f_hat = np.fft.fft2(f) * dealiasing_mask
    return np.real(np.fft.ifft2(f_hat))

def get_grad_dealiased(f):
    f_hat = np.fft.fft2(f) * dealiasing_mask
    return np.real(np.fft.ifft2(1j * KX * f_hat)), np.real(np.fft.ifft2(1j * KY * f_hat))

def get_laplacian_dealiased(f):
    f_hat = np.fft.fft2(f) * dealiasing_mask
    return np.real(np.fft.ifft2(-K2 * f_hat))

def compute_rhs(rho_in, phi_in):
    dphi_dx, dphi_dy = get_grad_dealiased(phi_in)
    grad_phi_sq = apply_filter(dphi_dx**2 + dphi_dy**2)
    grad_phi_sq = np.maximum(grad_phi_sq, 0.0)
    
    Gamma = gamma0 * 0.5 * (1.0 + np.tanh((grad_phi_sq - kc**2) / delta_k))
    Deff = np.maximum(D0 - (lc**2) * Gamma, 0.0)
    
    lap_phi = get_laplacian_dealiased(phi_in)
    rhs_phi = Deff * lap_phi - 0.5 * hbar_over_m * grad_phi_sq
    
    drho_dx, drho_dy = get_grad_dealiased(rho_in)
    vx, vy = hbar_over_m * dphi_dx, hbar_over_m * dphi_dy
    div_rho_v = apply_filter(drho_dx * vx + drho_dy * vy + rho_in * hbar_over_m * lap_phi)
    
    flux_x = Deff * drho_dx
    flux_y = Deff * drho_dy
    dflux_x_dx, _ = get_grad_dealiased(flux_x)
    _, dflux_y_dy = get_grad_dealiased(flux_y)
    
    rhs_rho = -div_rho_v + (dflux_x_dx + dflux_y_dy)
    return rhs_rho, rhs_phi, Deff

def count_localizations(rho_field, threshold_sigma=2.0, min_distance_px=4):
    mean_r, std_r = np.mean(rho_field), np.std(rho_field)
    thresh = mean_r + threshold_sigma * std_r
    local_max = (rho_field == maximum_filter(rho_field, size=min_distance_px))
    peaks = local_max & (rho_field > thresh)
    labeled, num_features = label(peaks)
    return num_features

def get_peak_positions(rho_field, threshold_sigma=2.0, min_distance_px=4):
    mean_r, std_r = np.mean(rho_field), np.std(rho_field)
    thresh = mean_r + threshold_sigma * std_r
    local_max = (rho_field == maximum_filter(rho_field, size=min_distance_px))
    peaks = local_max & (rho_field > thresh)
    y_pos, x_pos = np.where(peaks)
    return np.column_stack((x_pos, y_pos))

def track_peaks(positions_old, positions_new, max_distance=10):
    if len(positions_old) == 0 or len(positions_new) == 0:
        return {}, {}
    distances = cdist(positions_old, positions_new)
    matches = {}
    unmatched_new = set(range(len(positions_new)))
    for i in range(len(positions_old)):
        for j in range(len(positions_new)):
            if distances[i, j] < max_distance and j in unmatched_new:
                matches[i] = j
                unmatched_new.discard(j)
                break
    return matches, {'old': [], 'new': list(unmatched_new)}

def get_shape_eta(rho_field, pos, window=5):
    """Tepe civarındaki daireselliği hesaplar (eta = lambda_min / lambda_max)"""
    px, py = int(pos[0]), int(pos[1])
    Nx, Ny = rho_field.shape
    
    # Pencere sınırlarını kontrol et
    x0 = max(0, px - window//2)
    x1 = min(Nx, px + window//2 + 1)
    y0 = max(0, py - window//2)
    y1 = min(Ny, py + window//2 + 1)
    
    r_sub = rho_field[y0:y1, x0:x1]
    
    if r_sub.size < 4:
        return 1.0
    
    # Eylemsizlik momenti
    cy, cx_local = np.indices(r_sub.shape)
    cx_center = r_sub.shape[1] / 2
    cy_center = r_sub.shape[0] / 2
    
    Ixx = np.sum(r_sub * (cx_local - cx_center)**2)
    Iyy = np.sum(r_sub * (cy - cy_center)**2)
    Ixy = np.sum(r_sub * (cx_local - cx_center) * (cy - cy_center))
    
    cov = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    eigvals = np.linalg.eigvalsh(cov)
    
    if eigvals[1] > 0 and eigvals[0] >= 0:
        return eigvals[0] / eigvals[1]
    return 1.0

# ==========================================
# 3. A_PHI TARAMASI (ANA TARAMA)
# ==========================================
print("="*70)
print("📊 DİFÜZYON KİLİTLENMESİ - TAM ANALİZ (Kalıcılık + Form)")
print("="*70)

A_phi_list = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0]
results = {}
persistence_results = {}

print("\n[1/4] A_phi Taraması Başlatılıyor...")
print("-"*50)

for A_phi in A_phi_list:
    np.random.seed(42)
    rho = 1.0 + 0.01 * np.random.randn(N, N)
    rho = np.maximum(rho, 0.1)
    
    phi = A_phi * (np.sin(2 * np.pi * X / L) * np.cos(2 * np.pi * Y / L) + 
                   0.3 * np.sin(4 * np.pi * X / L) * np.cos(4 * np.pi * Y / L))
    phi += 0.02 * np.random.randn(N, N)
    
    t_hist, flock_hist, Nloc_hist, Deff_min_hist, grad_max_hist = [], [], [], [], []
    
    # Kalıcılık ve form için ek veriler
    persistence_hist = []
    stable_hist = []
    shape_hist = []
    tracker_data = {}  # {id: {'pos': (x,y), 'life': int, 'history': []}}
    next_id = 0
    K_threshold = 5  # Kalıcılık eşiği
    
    for step in range(total_steps):
        t = step * dt
        
        k1_r, k1_p, Deff_c = compute_rhs(rho, phi)
        r_pred = np.maximum(rho + dt * k1_r, 1e-4)
        p_pred = phi + dt * k1_p
        
        k2_r, k2_p, _ = compute_rhs(r_pred, p_pred)
        
        rho = np.maximum(rho + dt * 0.5 * (k1_r + k2_r), 1e-4)
        phi = phi + dt * 0.5 * (k1_p + k2_p)
        
        if np.isnan(rho).any() or np.isnan(phi).any():
            print(f"UYARI: NaN oluştu! A_phi={A_phi}, adım={step}")
            break
        
        if step % 50 == 0:
            _, _, Deff_curr = compute_rhs(rho, phi)
            
            dphi_dx, dphi_dy = get_grad_dealiased(phi)
            grad_phi_sq = apply_filter(dphi_dx**2 + dphi_dy**2)
            grad_phi_sq = np.maximum(grad_phi_sq, 0.0)
            max_grad = np.sqrt(np.max(grad_phi_sq))
            
            f_lock = np.sum(Deff_curr < 0.05) / (N * N)
            n_loc = count_localizations(rho)
            
            t_hist.append(t)
            flock_hist.append(f_lock)
            Nloc_hist.append(n_loc)
            Deff_min_hist.append(np.min(Deff_curr))
            grad_max_hist.append(max_grad)
            
            # ===== KALICILIK VE FORM ANALİZİ =====
            # Tepe konumlarını bul
            positions = get_peak_positions(rho)
            
            # Eşleştirme
            updated_trackers = {}
            matched_ids = set()
            
            for pos in positions:
                best_id = None
                min_dist = 5.0
                
                for tid, data in tracker_data.items():
                    if tid in matched_ids:
                        continue
                    lx, ly = data['pos']
                    dist = np.sqrt((pos[0] - lx)**2 + (pos[1] - ly)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_id = tid
                
                if best_id is not None:
                    matched_ids.add(best_id)
                    hist = tracker_data[best_id]['history']
                    hist.append(tuple(pos))
                    updated_trackers[best_id] = {
                        'pos': tuple(pos),
                        'life': tracker_data[best_id]['life'] + 1,
                        'history': hist
                    }
                else:
                    updated_trackers[next_id] = {
                        'pos': tuple(pos),
                        'life': 1,
                        'history': [tuple(pos)]
                    }
                    next_id += 1
            
            tracker_data = updated_trackers
            
            # Metrikler
            n_persistent = sum(1 for d in tracker_data.values() if d['life'] >= K_threshold)
            
            # Kararlılık (son 3 adımda hareket < 0.2 piksel)
            n_stable = 0
            shapes = []
            
            for tid, d in tracker_data.items():
                if d['life'] >= K_threshold:
                    if len(d['history']) >= 3:
                        h = np.array(d['history'][-3:])
                        dr = np.mean(np.sqrt(np.sum(np.diff(h, axis=0)**2, axis=1)))
                        if dr < 0.2:
                            n_stable += 1
                    
                    # Şekil daireselliği
                    eta = get_shape_eta(rho, d['pos'])
                    shapes.append(eta)
            
            persistence_hist.append(n_persistent)
            stable_hist.append(n_stable)
            shape_hist.append(np.mean(shapes) if shapes else 1.0)
            
    results[A_phi] = {
        't': t_hist,
        'f_lock': flock_hist,
        'N_loc': Nloc_hist,
        'min_Deff': Deff_min_hist,
        'max_grad': grad_max_hist,
        'rho_final': rho,
        'phi_final': phi
    }
    
    persistence_results[A_phi] = {
        'n_persistent': persistence_hist,
        'n_stable': stable_hist,
        'shape': shape_hist,
        'n_peaks': Nloc_hist  # N_loc ile aynı
    }
    
    print(f"A_phi = {A_phi:.1f} | min Deff: {Deff_min_hist[-1]:.4f} | f_lock: %{flock_hist[-1]*100:.1f} | N_loc: {Nloc_hist[-1]} | N_persistent: {persistence_hist[-1]}")

# ==========================================
# 4. DRIFT TAKİBİ (SADECE A_PHI=3.0 ve 4.0)
# ==========================================
print("\n[2/4] Drift Takibi Başlatılıyor...")
print("-"*50)

drift_results = {}

for A_phi in [3.0, 4.0]:
    np.random.seed(42)
    rho = 1.0 + 0.01 * np.random.randn(N, N)
    rho = np.maximum(rho, 0.1)
    
    phi = A_phi * (np.sin(2 * np.pi * X / L) * np.cos(2 * np.pi * Y / L) + 
                   0.3 * np.sin(4 * np.pi * X / L) * np.cos(4 * np.pi * Y / L))
    phi += 0.02 * np.random.randn(N, N)
    
    tracked_trajectories = defaultdict(list)
    all_positions = []
    
    positions = get_peak_positions(rho)
    all_positions.append(positions)
    for idx, pos in enumerate(positions):
        tracked_trajectories[idx].append(pos)
    
    next_id = len(positions)
    
    for step in range(1, total_steps + 1):
        k1_r, k1_p, _ = compute_rhs(rho, phi)
        r_pred = np.maximum(rho + dt * k1_r, 1e-4)
        p_pred = phi + dt * k1_p
        k2_r, k2_p, _ = compute_rhs(r_pred, p_pred)
        rho = np.maximum(rho + dt * 0.5 * (k1_r + k2_r), 1e-4)
        phi = phi + dt * 0.5 * (k1_p + k2_p)
        
        if step % 50 == 0:
            new_positions = get_peak_positions(rho)
            all_positions.append(new_positions)
            
            if len(all_positions) > 1:
                old_positions = all_positions[-2]
                if len(old_positions) > 0 and len(new_positions) > 0:
                    matches, _ = track_peaks(old_positions, new_positions)
                    
                    for old_idx, new_idx in matches.items():
                        old_id = None
                        for pid, traj in tracked_trajectories.items():
                            if len(traj) > 0 and np.array_equal(traj[-1], old_positions[old_idx]):
                                old_id = pid
                                break
                        if old_id is not None:
                            tracked_trajectories[old_id].append(new_positions[new_idx])
                        else:
                            tracked_trajectories[next_id] = [new_positions[new_idx]]
                            next_id += 1
                    
                    matched_new = [idx for _, idx in matches.items()]
                    for new_idx in range(len(new_positions)):
                        if new_idx not in matched_new:
                            tracked_trajectories[next_id] = [new_positions[new_idx]]
                            next_id += 1
    
    drift_results[A_phi] = {
        'trajectories': dict(tracked_trajectories),
        'final_peaks': len(new_positions) if 'new_positions' in locals() else 0
    }
    
    displacements = []
    for pid, traj in tracked_trajectories.items():
        if len(traj) > 1:
            traj_arr = np.array(traj)
            total_disp = np.sum(np.sqrt(np.diff(traj_arr[:, 0])**2 + np.diff(traj_arr[:, 1])**2)) * (L/N)
            displacements.append(total_disp)
    
    if displacements:
        print(f"A_phi = {A_phi:.1f} | Toplam tepe: {len(tracked_trajectories)} | Ort. drift: {np.mean(displacements):.2f} ± {np.std(displacements):.2f}")

# ==========================================
# 5. VERİLERİ TOPLA
# ==========================================
A_vals_full = np.array(A_phi_list)
deff_final = np.array([results[A]['min_Deff'][-1] for A in A_phi_list])
flock_final = np.array([results[A]['f_lock'][-1] * 100 for A in A_phi_list])
nloc_final = np.array([results[A]['N_loc'][-1] for A in A_phi_list])
grad_final = np.array([results[A]['max_grad'][-1] for A in A_phi_list])

# Persistence verileri
persistent_final = np.array([persistence_results[A]['n_persistent'][-1] for A in A_phi_list])
stable_final = np.array([persistence_results[A]['n_stable'][-1] for A in A_phi_list])
shape_final = np.array([persistence_results[A]['shape'][-1] for A in A_phi_list])

# Drift verileri
drift_mean = []
drift_std = []
drift_A_vals = []
for A in [3.0, 4.0]:
    if A in drift_results:
        displacements = []
        for pid, traj in drift_results[A]['trajectories'].items():
            if len(traj) > 1:
                traj_arr = np.array(traj)
                total_disp = np.sum(np.sqrt(np.diff(traj_arr[:, 0])**2 + np.diff(traj_arr[:, 1])**2)) * (L/N)
                displacements.append(total_disp)
        if displacements:
            drift_mean.append(np.mean(displacements))
            drift_std.append(np.std(displacements))
            drift_A_vals.append(A)

print("\n[3/4] Görselleştirme Başlatılıyor...")
print("-"*50)

# ==========================================
# 6. KALICILIK VE FORM GRAFİKLERİ
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. N_persistent zaman gelişimi (A_phi=4.0)
ax = axes[0, 0]
res_4 = persistence_results[4.0]
t_vals = results[4.0]['t']
ax.plot(t_vals, results[4.0]['N_loc'], 'r--', label=r'$N_{peaks}$ (Toplam)', linewidth=1.5)
ax.plot(t_vals, res_4['n_persistent'], 'g-', linewidth=2.5, label=r'$N_{persistent}$ (Kalıcı)')
ax.plot(t_vals, res_4['n_stable'], 'b-', linewidth=2.5, label=r'$N_{stable}$ (Kararlı)')
ax.set_xlabel('Zaman (t)')
ax.set_ylabel('Yapı Sayısı')
ax.set_title(r'$A_\phi=4.0$: Yapısal Ayrışma')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# 2. Shape (dairesellik) zaman gelişimi
ax = axes[0, 1]
for A in [1.6, 2.5, 3.0, 4.0]:
    if A in persistence_results:
        ax.plot(t_vals[:len(persistence_results[A]['shape'])], 
                persistence_results[A]['shape'], 
                label=f'$A_\\phi={A}$', linewidth=2)
ax.set_xlabel('Zaman (t)')
ax.set_ylabel(r'$\eta_{shape} = \lambda_{min}/\lambda_{max}$')
ax.set_title('Dairesellik (1.0 = Tam Küresel)')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.1)

# 3. N_persistent vs A_phi (son durum)
ax = axes[0, 2]
ax.plot(A_vals_full, nloc_final, 'ro-', linewidth=2, markersize=8, label='N_loc')
ax.plot(A_vals_full, persistent_final, 'go-', linewidth=2, markersize=8, label='N_persistent')
ax.plot(A_vals_full, stable_final, 'bo-', linewidth=2, markersize=8, label='N_stable')
ax.set_xlabel(r'$A_\phi$')
ax.set_ylabel('Yapı Sayısı')
ax.set_title('Son Durum Karşılaştırması')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# 4. Shape vs A_phi
ax = axes[1, 0]
ax.plot(A_vals_full, shape_final, 'mo-', linewidth=2, markersize=8)
ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Küresel (1.0)')
ax.set_xlabel(r'$A_\phi$')
ax.set_ylabel(r'$\eta_{shape}$ (son durum)')
ax.set_title('Dairesellik vs Başlangıç Genliği')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# 5. N_persistent vs Deff
ax = axes[1, 1]
ax.scatter(deff_final, persistent_final, c=A_vals_full, cmap='viridis', 
           s=150, edgecolors='k', linewidth=1.5)
for i, A in enumerate(A_vals_full):
    ax.annotate(f'{A:.1f}', (deff_final[i], persistent_final[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax.set_xlabel(r'$\min D_{eff}$')
ax.set_ylabel(r'$N_{persistent}$')
ax.set_title('Kalıcılık vs Difüzyon')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label(r'$A_\phi$')

# 6. Shape vs Deff
ax = axes[1, 2]
ax.scatter(deff_final, shape_final, c=A_vals_full, cmap='plasma', 
           s=150, edgecolors='k', linewidth=1.5)
for i, A in enumerate(A_vals_full):
    ax.annotate(f'{A:.1f}', (deff_final[i], shape_final[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax.set_xlabel(r'$\min D_{eff}$')
ax.set_ylabel(r'$\eta_{shape}$')
ax.set_title('Dairesellik vs Difüzyon')
ax.grid(True, alpha=0.3)
cbar2 = plt.colorbar(ax.collections[0], ax=ax)
cbar2.set_label(r'$A_\phi$')

plt.tight_layout()
plt.savefig('kalicilik_ve_form_analizi.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 7. KAPSAMLI GRAFİK (ÖNCEKİ GİBİ)
# ==========================================
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

# 1. min Deff
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(A_vals_full, deff_final, 'bo-', linewidth=2.5, markersize=10)
ax1.fill_between(A_vals_full, 0, deff_final, alpha=0.2, color='blue')
ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel(r'$A_\phi$', fontsize=12)
ax1.set_ylabel(r'$\min D_{eff}$', fontsize=12)
ax1.set_title('Difüzyon Kilitlenmesi', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.05)

# 2. f_lock
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(A_vals_full, flock_final, 'ro-', linewidth=2.5, markersize=10)
ax2.fill_between(A_vals_full, 0, flock_final, alpha=0.2, color='red')
ax2.set_xlabel(r'$A_\phi$', fontsize=12)
ax2.set_ylabel(r'$f_{lock}$ (%)', fontsize=12)
ax2.set_title('Kilitlenmiş Hacim Oranı', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. N_loc
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(A_vals_full, nloc_final, 'go-', linewidth=2.5, markersize=10)
ax3.fill_between(A_vals_full, 0, nloc_final, alpha=0.2, color='green')
ax3.set_xlabel(r'$A_\phi$', fontsize=12)
ax3.set_ylabel(r'$N_{loc}$', fontsize=12)
ax3.set_title('Lokalize Yapı Sayısı', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. max_grad
ax4 = fig.add_subplot(gs[0, 3])
ax4.plot(A_vals_full, grad_final, 'mo-', linewidth=2.5, markersize=10)
ax4.axhline(kc, color='k', linestyle='--', alpha=0.5, label=f'k_c={kc}')
ax4.fill_between(A_vals_full, 0, grad_final, alpha=0.2, color='magenta')
ax4.set_xlabel(r'$A_\phi$', fontsize=12)
ax4.set_ylabel(r'$\max|\nabla\phi|$', fontsize=12)
ax4.set_title('Faz Eğimi', fontsize=13, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. N_persistent (yeni)
ax5 = fig.add_subplot(gs[1, 0:2])
ax5.plot(A_vals_full, persistent_final, 'go-', linewidth=2.5, markersize=10, label='N_persistent')
ax5.plot(A_vals_full, stable_final, 'bo-', linewidth=2.5, markersize=10, label='N_stable')
ax5.set_xlabel(r'$A_\phi$', fontsize=12)
ax5.set_ylabel('Yapı Sayısı', fontsize=12)
ax5.set_title('Kalıcı ve Kararlı Yapılar', fontsize=13, fontweight='bold')
ax5.legend(loc='best', fontsize=10)
ax5.grid(True, alpha=0.3)

# 6. Shape (yeni)
ax6 = fig.add_subplot(gs[1, 2:4])
ax6.plot(A_vals_full, shape_final, 'mo-', linewidth=2.5, markersize=10)
ax6.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Küresel')
ax6.set_xlabel(r'$A_\phi$', fontsize=12)
ax6.set_ylabel(r'$\eta_{shape}$', fontsize=12)
ax6.set_title('Dairesellik (1.0 = Tam Küresel)', fontsize=13, fontweight='bold')
ax6.legend(loc='best', fontsize=10)
ax6.grid(True, alpha=0.3)

# 7. Drift
ax7 = fig.add_subplot(gs[2, 0:2])
if drift_mean:
    ax7.errorbar(drift_A_vals, drift_mean, yerr=drift_std, 
                fmt='o-', color='purple', linewidth=2.5, markersize=12,
                capsize=8, elinewidth=2, capthick=2)
    ax7.set_xlabel(r'$A_\phi$', fontsize=12)
    ax7.set_ylabel('Ortalama Yer Değiştirme', fontsize=12)
    ax7.set_title('Drift Analizi (A$_\\phi$=3.0, 4.0)', fontsize=13, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    for i, A in enumerate(drift_A_vals):
        ax7.text(A, drift_mean[i] + drift_std[i] + 2, 
                f'{drift_mean[i]:.2f}±{drift_std[i]:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# 8. Özet bilgi
ax8 = fig.add_subplot(gs[2, 2:4])
ax8.axis('off')
corr_matrix = np.corrcoef(nloc_final, deff_final)
r_squared = corr_matrix[0, 1]**2

info_text = f"""
📊 ÖZET BİLGİLER

🔑 Kritik Eşik:
   A_phi ≈ 2.0

📈 Yapı Metrikleri:
   N_loc Max: {np.max(nloc_final)}
   N_persistent Max: {np.max(persistent_final)}
   N_stable Max: {np.max(stable_final)}

🔬 Dairesellik:
   Min: {np.min(shape_final):.3f}
   Max: {np.max(shape_final):.3f}
   (1.0 = Tam Küresel)

📌 Faz Bölgeleri:
   I: A<1.6 (Aktif)
   II: 1.6-2.5 (Geçiş)
   III: A>2.5 (Kilitli)

💡 R² (N_loc vs Deff): 
   {r_squared:.3f}
"""
ax8.text(0.1, 0.9, info_text, transform=ax8.transAxes,
        fontsize=11, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('DİFÜZYON KİLİTLENMESİ - KAPSAMLI ANALİZ (Kalıcılık + Form)', 
            fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('kapsamli_analiz_persistence.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 8. ZAMAN GELİŞİMİ GRAFİKLERİ
# ==========================================
selected_A = [0.8, 1.6, 2.0, 3.0, 4.0]
colors = ['blue', 'green', 'orange', 'red', 'purple']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# f_lock
ax = axes[0, 0]
for i, A in enumerate(selected_A):
    if A in results and results[A]['t']:
        ax.plot(results[A]['t'], np.array(results[A]['f_lock']) * 100,
                color=colors[i], label=f'A$_\\phi$={A}', linewidth=2)
ax.set_xlabel('Zaman (t)')
ax.set_ylabel(r'$f_{lock}$ (%)')
ax.set_title('Kilitlenmiş Hacim Oranı')
ax.legend()
ax.grid(True)

# N_loc
ax = axes[0, 1]
for i, A in enumerate(selected_A):
    if A in results and results[A]['t']:
        ax.plot(results[A]['t'], results[A]['N_loc'],
                color=colors[i], label=f'A$_\\phi$={A}', linewidth=2)
ax.set_xlabel('Zaman (t)')
ax.set_ylabel(r'$N_{loc}$')
ax.set_title('Lokalize Yapı Sayısı')
ax.legend()
ax.grid(True)

# N_persistent (yeni)
ax = axes[1, 0]
for i, A in enumerate(selected_A):
    if A in persistence_results and results[A]['t']:
        ax.plot(results[A]['t'][:len(persistence_results[A]['n_persistent'])], 
                persistence_results[A]['n_persistent'],
                color=colors[i], label=f'A$_\\phi$={A}', linewidth=2)
ax.set_xlabel('Zaman (t)')
ax.set_ylabel(r'$N_{persistent}$')
ax.set_title('Kalıcı Yapı Sayısı')
ax.legend()
ax.grid(True)

# Shape (yeni)
ax = axes[1, 1]
for i, A in enumerate([1.6, 2.0, 3.0, 4.0]):
    if A in persistence_results and results[A]['t']:
        ax.plot(results[A]['t'][:len(persistence_results[A]['shape'])], 
                persistence_results[A]['shape'],
                color=colors[i+1], label=f'A$_\\phi$={A}', linewidth=2)
ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Küresel')
ax.set_xlabel('Zaman (t)')
ax.set_ylabel(r'$\eta_{shape}$')
ax.set_title('Dairesellik Gelişimi')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('zaman_gelisimi_persistence.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 9. ÖZET RAPORU
# ==========================================
print("\n" + "="*90)
print("📊 DİFÜZYON KİLİTLENMESİ - ÖZET RAPORU (Kalıcılık + Form)")
print("="*90)

print("\n🔑 KRİTİK PARAMETRELER:")
print("-"*50)
print(f"  • Kritik A_phi eşiği: ~2.0 (Deff → 0)")
print(f"  • Maksimum N_loc: {np.max(nloc_final)} (A_phi = {A_vals_full[np.argmax(nloc_final)]})")
print(f"  • Maksimum N_persistent: {np.max(persistent_final)} (A_phi = {A_vals_full[np.argmax(persistent_final)]})")
print(f"  • Maksimum N_stable: {np.max(stable_final)} (A_phi = {A_vals_full[np.argmax(stable_final)]})")

print("\n📈 DAİRESELLİK:")
print("-"*50)
print(f"  • Minimum η_shape: {np.min(shape_final):.3f} (A_phi = {A_vals_full[np.argmin(shape_final)]})")
print(f"  • Maksimum η_shape: {np.max(shape_final):.3f} (A_phi = {A_vals_full[np.argmax(shape_final)]})")
print(f"  • (η=1.0 = Tam Küresel)")

print("\n📌 FAZ GEÇİŞİ:")
print("-"*50)
print("  Bölge I (A_phi < 1.6):  Difüzyon aktif, yapılar dağınık")
print("  Bölge II (1.6 < A_phi < 2.5): Geçiş, yapılar çöküyor")
print("  Bölge III (A_phi > 2.5): Kilitlenme, yapılar yeniden oluşuyor")

if drift_mean:
    print("\n🔬 LOKALİZASYON-DRIFT İLİŞKİSİ:")
    print("-"*50)
    print(f"  • A_phi=3.0: N_loc={nloc_final[5]}, N_persistent={persistent_final[5]}, Drift={drift_mean[0]:.2f}±{drift_std[0]:.2f}")
    print(f"  • A_phi=4.0: N_loc={nloc_final[6]}, N_persistent={persistent_final[6]}, Drift={drift_mean[1]:.2f}±{drift_std[1]:.2f}")

print("\n" + "="*90)
print("✅ Analiz tamamlandı!")
print(f"   • Kaydedilen dosyalar: kapsamli_analiz_persistence.png, zaman_gelisimi_persistence.png, kalicilik_ve_form_analizi.png")