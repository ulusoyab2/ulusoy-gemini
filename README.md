# Ulusoy-Gemini V3: Self-Induced Anderson Localization in Vacuum Phase Dynamics

## Repository Structure

### Core Simulation Scripts

1. **`diffusion_localization_2d.py`**  
   *2D Diffusion Collapse and Density Localization*  
   - Simulates phase dynamics in 2D with effective diffusion coefficient \(D_{\mathrm{eff}}\)  
   - Demonstrates \(D_{\mathrm{eff}} \to 0\) threshold behavior  
   - Analyzes density islands (\(\rho\)-islands) formation and persistence  
   - Generates comprehensive localization metrics (\(N_{loc}\), \(f_{lock}\), shape analysis)

2. **`structure_analysis_3d.py`**  
   *3D Simulations and Stable Structure Clustering*  
   - Extends analysis to 3D physical space  
   - Performs DBSCAN clustering on final stable structures  
   - Extracts structure properties: energy, effective radius, anisotropy  
   - Validates universality of localization independent of initial conditions

3. **`topological_quantization.py`**  
   *Topological Winding Number Quantization Dynamics*  
   - Computes topological winding number \(w\) via holonomy on \(S^2\)  
   - Demonstrates relaxation from arbitrary \(w\) to quantized values (\(0, 0.5, 1.0\))  
   - Provides numerical proof of SU(2) spinor quantization  
   - Shows topological filtering mechanism in vacuum dynamics

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analyses
python src/diffusion_localization_2d.py
python src/structure_analysis_3d.py
python src/topological_quantization.py
