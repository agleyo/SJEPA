# SJEPA


## Installation

```bash
# 1. Récupérer le projet
git clone <https://github.com/agleyo/SJEPA.git> sjepa
cd sjepa

# 2. Environnement virtuel
python -m venv .venv
source .venv/bin/activate            # Linux / macOS
# .venv\Scripts\activate             # Windows (cmd)
# .\.venv\Scripts\Activate.ps1       # Windows (PowerShell)

# 3. Dépendances (CPU)
pip install --upgrade pip
pip install torch torchvision pillow numpy matplotlib
```

**GPU CUDA 12.1 :**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**GPU CUDA 11.8 :**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Démarrage rapide
```
python main.py --skip-baselines --arms vicreg_hyb --epochs 300
python plot_probes.py
```
