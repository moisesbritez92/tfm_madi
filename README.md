# Comparación de Codificadores Visuales en Diffusion Policy: Push-T

## Resumen

Este trabajo compara cinco codificadores visuales (backbones) como observadores (`obs_encoder`) en una Diffusion Policy para la tarea de manipulación robótica **Push-T**. El objetivo es evaluar el impacto del tipo de codificador, su escala y su estrategia de entrenamiento (congelado vs. ajuste fino) en el desempeño, eficiencia muestral y coste computacional.

## Motivación

Las políticas de difusión han mostrado resultados prometedores en manipulación robótica, pero la arquitectura del codificador visual es un componente crítico que afecta tanto al desempeño como al coste computacional. Esta comparación establece una línea base para futuras decisiones de arquitectura.

## Variantes Experimentales

Se entrenan cinco modelos sobre el mismo dataset de Push-T (206 demostraciones humanas):

| Variante | Backbone | Estrategia | Desempeño Test | Minutos/Época |
|---|---|---|---|---|
| **V0** | ResNet-18 | Entrenado desde cero | **0,8645** | 1,5 |
| **V1** | ResNet-18 ImageNet | Congelado | 0,668 | 5,3 |
| **V2** | ResNet-18 ImageNet | Ajuste fino | 0,6477 | 14,7 |
| **V3** | DINOv2 ViT-S/14 | Congelado | 0,6224 | — |
| **V4** | CLIP ViT-B/16 | Congelado | 0,5351 | — |

**Resultado final** (prueba disjunta `200000–200199`, 200 episodios): V0 alcanza **0,872**, validando que el entrenamiento desde cero con datos de tarea supera al preentrenamiento visual en este dominio.

## Estructura del Repositorio

```
.
├── CLAUDE.md                          # Contexto operativo del proyecto
├── memoria/                           # Documento LaTeX del TFM
│   ├── main.tex
│   ├── secciones/                     # Capítulos: introducción, estado del arte, etc.
│   ├── bib/                           # Referencias bibliográficas
│   └── scripts/                       # Análisis y generación de figuras
├── diffuser/
│   ├── experimento_encoder_pusht.md   # Diseño experimental detallado
│   ├── scripts/
│   │   ├── run_encoder_exp.sh         # Script de entrenamiento (usa Hydra)
│   │   ├── evaluar_bloque_test.py     # Evaluación en bloque disjunto
│   │   └── caracterizar_dataset.py    # Análisis del dataset
│   ├── v{0..4}_inference_utils.py     # Utilidades de inferencia para cada variante
│   ├── inferencia_v{0..4}_pusht.ipynb # Notebooks de demostración
│   ├── models/                        # Checkpoints congelados (60 GB, gitignored)
│   │   └── V{0..4}/
│   ├── godot/                         # Integración con simulador Godot 4.7.2
│   └── artifacts/                     # Salidas de inference
├── logs_entrenamiento/
│   ├── resumen.json                   # Métricas consolidadas
│   ├── v{0..4}_seed42_epocas.csv      # Timings por época
│   ├── prueba_final/                  # JSON de resultados preregistrados
│   └── godot_paper/                   # Contraste con modelo original
└── papers/                            # Referencias externas

```

## Configuración de Entrenamientos

- **Dataset**: 90 episodios (train) + 4 (val), seeds `100000–100049` (test)
- **Partición**: fija por `task.seed: 42`, determinista entre variantes
- **Presupuesto**: 500 épocas máximo con parada anticipada (V0, V1 → 500 épocas; V2 → 266)
- **Hardware**: RTX 3070 Ti (8 GB VRAM) + WSL2 Ubuntu 24.04
- **Framework**: Hydra 1.2.0, PyTorch 1.12.1+cu116, diffusers 0.11.1

### Overrides de Hydra

```bash
dataloader.num_workers=2
dataloader.persistent_workers=false
val_dataloader.num_workers=0
task.env_runner.n_envs=8
logging.mode=disabled
```

## Resultados Principales

### Prueba Disjunta Final (26–27/08/2026)

Protocolo preregistrado: 5 variantes × 200 episodios nuevos (seeds `200000–200199`), una sola realización de ruido de difusión.

```
V0:  0,872 (50/200 éxito)
V1:  0,649
V2:  0,586
V3:  0,578
V4:  0,490
```

### Análisis de Coste

El coste por época revela un factor 9,8 entre V0 y V2:
- V0 (scratch): 1,5 min
- V1 (frozen): 5,3 min
- V2 (fine-tune): 14,7 min

El preentrenamiento no compensa la pérdida de desempeño. La mejor relación coste-beneficio es **V0**.

## Cómo Empezar

### Requisitos

- Python 3.9+
- PyTorch 1.12.1 + CUDA 11.6
- Diffusers 0.11.1
- timm 0.9.16

### Instalación (WSL2)

```bash
conda create -n robodiff python=3.9.15
conda activate robodiff
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
pip install diffusers==0.11.1 timm==0.9.16 hydra-core==1.2.0 zarr==2.12.0
pip install robomimic==0.2.0 --no-deps
```

### Entrenamiento

```bash
cd ~/tfm/diffusion_policy
./run_encoder_exp.sh v0 42              # Entrenar V0, seed 42
./run_encoder_exp.sh v0 42 --resume     # Reanudar desde checkpoint
./run_encoder_exp.sh --status           # Ver estado
./run_encoder_exp.sh --stop             # Detener gracefully
```

### Inferencia (Windows)

```python
import torch
from diffuser.v0_inference_utils import load_policy_bundle

# Cargar política congelada
device = torch.device('cuda')
policy, stats = load_policy_bundle('diffuser/models/V0/epoch=0350-test_mean_score=0.8645.ckpt')
policy = policy.to(device).eval()

# Ver notebooks diffuser/inferencia_v0_pusht.ipynb
```

## Notas Importantes

1. **No confundir directorios**: el entrenamiento ocurre en WSL (`~/tfm/diffusion_policy`), los checkpoints se copian a Windows (`diffuser/models/V*/`).

2. **Dataset en memoria**: El buffer de reproducción ocupa **2,84 GB en RAM**. Abrir en lectura con `zarr.open(..., 'r')` para análisis sin ese coste.

3. **Reanudar V3/V4 requiere parche**: El scheduler de learning rate se debe reconstruir correctamente; ver `CLAUDE.md` para detalles del bug del 28/08/2026.

4. **Modelo de referencia** (`V_Paper`): El artículo original usa spatial softmax de robomimic. Nuestro V0 no lo usa; la comparación se hace en `logs_entrenamiento/prueba_final/`.

5. **Validación en Godot**: Disponible en `diffuser/godot/` con dos modos de observación; **no reportable**, solo demostrativo para defensa.

## Referencias

- Chi, C., et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. ICML.
- Configuración base: `diffusion_policy/config/pusht_image.yaml`
- Detalles experimentales: [diffuser/experimento_encoder_pusht.md](diffuser/experimento_encoder_pusht.md)

## Contacto

- Autor: Moises Britez
- Director: Diego Borro
- Centro: Tecnun, Universidad de Navarra (MADI, 2026)

## GitHub
https://github.com/moisesbritez92/tfm_madi.git
