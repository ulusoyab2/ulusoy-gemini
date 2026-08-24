import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import label
from sklearn.preprocessing import StandardScaler
import sklearn.cluster as cluster
import time

# ==============================================================================
# 1. SIMÜLASYON FONKSİYONLARI (Önceki haliyle)
# ==============================================================================
def hyperbolic_decay(t, A0, Gamma):
    return 1.0 + (A0 - 1.0) / (1.0 + Gamma * t)

def compute_anisotropy_3d(phi, dx):
    grad_x, grad_y, grad_z = np.gradient(phi, dx, edge_order=2)
    I_x = np.sum(grad_x**2)
    I_y = np.sum(grad_y**2)
    I_z = np.sum(grad_z**2)
    I_min = min(I_x, I_y, I_z)
    if I_min < 1e-12:
        return 1.0, 1.0
    A_t = max(I_x, I_y, I_z) / I_min
    bc_ratio = I_y / I_z if I_z > 1e-12 else 1.0
    return A_t, bc_ratio

# ==============================================================================
# 2. YENİ MODÜL: KARARLI YAPI ANALİZİ (Çıktı Tablosu)
# ==============================================================================
def extract_pure_outputs(phi, dx):
    """
    BAŞLANGIÇ BİLGİSİ (INIT) İÇERMEZ.
    Sadece t_final anındaki phi matrisinden kararlı yapıları türetir.
    """
    threshold = 0.1
    mask = phi > threshold
    labeled_array, num_features = label(mask)
    
    if num_features == 0:
        return []

    stable_structures = []

    for i in range(1, num_features + 1):
        p_mask = (labeled_array == i)
        coords = np.argwhere(p_mask)
        
        # Gürültü filtresi (çok küçük sayısal dalgalanmaları ele)
        if len(coords) < 8:
            continue

        # 1. Tepe Genliği
        phi_max = float(np.max(phi[p_mask]))

        # 2. Alan İntegrali
        I_phi = float(np.sum(phi[p_mask]) * (dx**3))

        # 3. Bileşen Enerjisi
        gx, gy, gz = np.gradient(phi, dx)
        grad_sq = gx**2 + gy**2 + gz**2
        V_phi = 0.25 * (phi**2 - 1.0)**2
        E_comp = float(np.sum((0.5 * grad_sq + V_phi)[p_mask]) * (dx**3))

        # 4. Boyutsal Olarak Doğru Karakteristik Yarıçap (Reff)
        cov = np.cov(coords.T)
        eigvals = np.linalg.eigvalsh(cov)
        R_eff = float(np.sqrt(np.sum(np.maximum(eigvals, 0.0))) * dx)

        # 5. Geometrik Anizotropi (Acov)
        e_sorted = np.sort(np.maximum(eigvals, 1e-10))[::-1]
        A_cov = float(np.sqrt(e_sorted[0] / e_sorted[2]))

        stable_structures.append({
            'E_comp': E_comp,
            'R_eff': R_eff,
            'phi_max': phi_max,
            'I_phi': I_phi,
            'A_cov': A_cov
        })

    return stable_structures


def analyze_final_states(simulation_runs, dx):
    """
    Tüm simülasyon çıktılarından gelen kararlı yapıları toplar,
    z-score normalizasyonu yapar ve doğal kümeleri arar.
    """
    all_structures = []
    summary_table = []

    for sim_id, phi_final in enumerate(simulation_runs):
        structures = extract_pure_outputs(phi_final, dx)
        N_stable = len(structures)
        
        if N_stable == 0:
            summary_table.append({
                'Sim': f"S{sim_id}",
                'N_stable': 0,
                'E_comp': "—",
                'R_eff': "—",
                'phi_max': "—",
                'A_cov': "—",
                'Durum': "Çözüldü / Vakum"
            })
        else:
            avg_E = np.mean([s['E_comp'] for s in structures])
            avg_R = np.mean([s['R_eff'] for s in structures])
            avg_phi = np.mean([s['phi_max'] for s in structures])
            avg_A = np.mean([s['A_cov'] for s in structures])
            
            summary_table.append({
                'Sim': f"S{sim_id}",
                'N_stable': N_stable,
                'E_comp': f"{avg_E:.3f}",
                'R_eff': f"{avg_R:.3f}",
                'phi_max': f"{avg_phi:.3f}",
                'A_cov': f"{avg_A:.4f}",
                'Durum': f"Kararlı (N={N_stable})"
            })
            all_structures.extend(structures)

    # NİHAİ ÇIKTI TABLOSU
    print("\n" + "="*85)
    print("SADECE ÇIKTI TABLOSU (INIT PARAMETRELERİ TAMAMEN ÇIKARILDI)")
    print("="*85)
    print(f"{'Sim':<6} | {'N_stable':<8} | {'<E_comp>':<10} | {'<R_eff>':<10} | {'<phi_max>':<10} | {'<A_cov>':<10} | {'Durum'}")
    print("-" * 85)
    for r in summary_table:
        print(f"{r['Sim']:<6} | {r['N_stable']:<8} | {str(r['E_comp']):<10} | {str(r['R_eff']):<10} | {str(r['phi_max']):<10} | {str(r['A_cov']):<10} | {r['Durum']}")
    print("-" * 85)

    # ÖLÇEKLENDİRİLMİŞ (KÖR) DBSCAN KÜMELEME
    if len(all_structures) >= 2:
        X = np.array([[s['E_comp'], s['R_eff'], s['phi_max'], s['I_phi']] for s in all_structures])
        X_scaled = StandardScaler().fit_transform(X)
        
        db = cluster.DBSCAN(eps=0.5, min_samples=2).fit(X_scaled)
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        
        print(f"\n[Sonuç] Başlangıç koşullarından tamamen bağımsız tespit edilen küme sayısı: {n_clusters}")

# ==============================================================================
# 3. 3B SİMÜLASYON (Çıktı Analizi Entegre Edildi)
# ==============================================================================
def run_3d_simulation_with_analysis(
    N_list=[96, 128, 160],
    L=10.0,
    dt=0.0004,
    n_steps=30000,
    save_every=100
):
    regimes = [
        (2.5, 1.0, 1.0, "Orta"),
        (1.5, 1.0, 1.0, "Deep")
    ]

    final_phi_list = []  # <-- Çıktı analizi için final yapıları saklar

    for N in N_list:
        dx = L / (N - 1)

        x = np.linspace(-L/2, L/2, N, dtype=np.float64)
        y = np.linspace(-L/2, L/2, N, dtype=np.float64)
        z = np.linspace(-L/2, L/2, N, dtype=np.float64)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        print("\n" + "="*70)
        print(f"3B SİMÜLASYON BAŞLATILDI: {N}^3 GRID")
        print(f"Grid aralığı dx = {dx:.6f} | dt = {dt} | Adım Sayısı = {n_steps}")
        print(f"Toplam Fiziksel Süre T = {dt*n_steps:.4f}")
        print("="*70)

        for sx, sy, sz, regime_name in regimes:
            t_start = time.time()

            phi = np.exp(-(X**2/(2*sx**2) + Y**2/(2*sy**2) + Z**2/(2*sz**2)))

            for step in range(1, n_steps + 1):
                lap = (np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
                       np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) +
                       np.roll(phi, 1, axis=2) + np.roll(phi, -1, axis=2) - 6.0 * phi) / (dx**2)
                phi += dt * (lap + (phi - phi**3))

            # Simülasyon sonundaki phi'yi listeye ekle
            final_phi_list.append(phi)

            t_elapsed = time.time() - t_start
            print(f"\n[{regime_name} Rejimi] (Süre: {t_elapsed:.1f} sn)")

    # ==========================================================================
    # 4. SİMÜLASYON BİTTİKTEN SONRA ÇIKTI ANALİZİNİ ÇALIŞTIR
    # ==========================================================================
    print("\n" + "="*70)
    print("SON DURUM KARARLI YAPI ANALİZİ BAŞLATILIYOR")
    print("="*70)
    analyze_final_states(final_phi_list, dx)


# ==============================================================================
# 5. BAŞLAT
# ==============================================================================
run_3d_simulation_with_analysis(
    N_list=[96, 128, 160],
    L=10.0,
    dt=0.0004,
    n_steps=30000,
    save_every=100
)