# Memoria del TFM en LaTeX

Plantilla ajustada a `normas_redaccion.md` (normas oficiales del centro), modelada sobre
el formato de `Vera Aguinaga, Jorge_TFM_MADI.pdf`, con citas en IEEE.

## Compilar

```bash
./compilar.sh      # o: make
make clean         # borra auxiliares
```

Genera `main.pdf`. El script ejecuta `pdflatex → bibtex main → bibtex con → pdflatex ×2`;
son necesarias las dos pasadas de bibtex porque la memoria lleva dos listas de obras.

## Estructura

```
memoria/
├── main.tex                 # datos de la memoria + orden de los capitulos
├── tecnun-tfm.sty           # estilo: caja, tipografia, encabezados, portada, indices
├── compilar.sh · Makefile
├── bst/
│   ├── IEEEtran.bst         # estilo IEEE, orden de aparicion  → Referencias
│   └── IEEEtranS.bst        # estilo IEEE, orden alfabetico    → Bibliografia
├── bib/
│   ├── referencias.bib      # obras CITADAS en el texto
│   └── bibliografia.bib     # obras CONSULTADAS no citadas
├── secciones/
│   ├── primera-hoja.tex · agradecimientos.tex · resumen.tex · abstract.tex
│   ├── 01-estado-del-arte.tex
│   ├── 02-metodologia.tex
│   ├── 03-resultados.tex
│   ├── 04-conclusiones.tex
│   ├── 05-presupuesto.tex
│   └── anexos.tex
└── img/                     # figuras; aqui va tambien logo-tecnun.pdf
```

Orden del documento, segun las normas: portada · primera hoja · agradecimientos · indice
de contenidos · indice de figuras · indice de tablas · resumen · abstract · **1 Estado del
arte · 2 Metodologia · 3 Resultados · 4 Conclusiones · 5 Presupuesto** · Referencias ·
Bibliografia · Anexos (A, B, …).

## Que impone el estilo (normas oficiales)

| Norma | Como se aplica |
|---|---|
| DIN A4; margenes 35 / 30 / 30 / 25 mm | `geometry` en `tecnun-tfm.sty` |
| Arial o similar, 11 pt, justificado, interlineado sencillo | Helvetica sin escalar, clon metrico de Arial |
| Capitulo: 16 pt, mayusculas, negrita, siempre en pagina impar | `titlesec` + opcion `openright` |
| Apartado: 14 pt, negrita, subrayado, minusculas | `\section` con `\uline` |
| Subapartado: 13 pt, negrita, subrayado | `\subsection` |
| Numeracion de apartados sin punto (`1.1 Titulo`) | formato de `titlesec` y del indice |
| Division menor sin numerar ni indexar | `\divisionmenor{Titulo}` |
| Encabezado par: titulo abreviado a la izquierda | `fancyhdr` |
| Encabezado impar: nombre del alumno a la derecha | `fancyhdr` |
| Pie centrado, «Pagina X de Y» | `\piepagina` con `lastpage` |
| Figuras y tablas centradas, con pie explicativo e indice propio | `caption`, `\listoffigures`, `\listoftables` |

Los preliminares van en numeros romanos con solo el numero al pie; el cuerpo, en arabigos
con «Pagina X de Y». Los capitulos caen siempre en pagina impar y el verso de relleno sale
sin encabezado ni numero.

**Dos divergencias conscientes respecto a las normas**, ambas pedidas expresamente: existe
una seccion *Bibliografia* (obras consultadas no citadas) que las normas no contemplan, y
figuras y tablas se numeran de forma corrida en todo el documento, como en el TFM modelo.

## Scripts de analisis y medida

`scripts/` produce los CSV de `datos/` que respaldan las tablas del capitulo de
resultados. Ninguna cifra de la memoria se escribe a mano.

| Script | Salida | Donde se ejecuta |
|---|---|---|
| `scripts/resumen_entrenamiento.py` | `../logs_entrenamiento/resumen.json` y CSV de tiempos | Windows |
| `scripts/analisis_dispersion.py` | `datos/dispersion_puntuaciones.csv`, `datos/wilcoxon_puntuaciones.csv` | Windows |
| `scripts/coste_parada_uniforme.py` | contrafactual de la parada anticipada (salida por pantalla) | Windows |
| `scripts/latencia_inferencia.py` | `datos/latencia_inferencia.csv` | Windows, `.venv_diffuser_infer`, **con GPU libre** |
| `../diffuser/scripts/memoria_gpu.py` | `datos/memoria_gpu.csv` | **WSL**, entorno `robodiff`, **con GPU libre** |

Los dos ultimos miden sobre la GPU y exigen que no haya nada mas usandola. La latencia se
mide en el entorno de inferencia de Windows; el pico de memoria, en el entorno de WSL en
el que se entreno, porque el asignador de PyTorch cambio entre las dos versiones y una
cifra tomada en Windows no describiria los entrenamientos de la Tabla 7.

```bash
# latencia: ronda las cinco variantes, ~45 min
.venv_diffuser_infer/Scripts/python.exe memoria/scripts/latencia_inferencia.py

# pico de VRAM: un proceso por variante y modo, desde WSL
wsl -d Ubuntu -- bash /mnt/c/Users/moise/Documents/0001_MADI/TFM/diffuser/scripts/medir_memoria_gpu.sh
```

## Citas IEEE

```latex
\cite{chi2025diffusion}                      → [1]
\cite{ho2020ddpm}, \cite{song2021sde}        → [2], [3]
\cite{song2019ncsn}--\cite{ross2011dagger}   → [4]-[5]
Chi \textit{et al.} \cite{chi2025diffusion}  → autor como sujeto de la frase
```

La referencia va **antes** del punto: `… reduce el error \cite{ho2020ddpm}.`
La numeracion la asigna BibTeX por orden de primera aparicion; no se toca a mano.

**Referencias** lista solo lo citado, numerado. **Bibliografia** lista lo consultado y no
citado, sin numero y ordenado alfabeticamente, para que no se confunda con las citas.
Si una obra de `bibliografia.bib` pasa a citarse, hay que moverla a `referencias.bib`.

## Claves disponibles en `bib/referencias.bib`

| Clave | Obra |
|---|---|
| `chi2025diffusion` | Chi *et al.*, Diffusion Policy, IJRR 44(10-11), 2025 |
| `chi2023diffusion` | Version de congreso del anterior, RSS 2023 |
| `ho2020ddpm` | Ho, Jain, Abbeel, DDPM, NeurIPS 2020 |
| `song2019ncsn` | Song, Ermon, score matching, NeurIPS 2019 |
| `song2021sde` | Song *et al.*, SDE, ICLR 2021 |
| `florence2021ibc` | Florence *et al.*, Implicit Behavioral Cloning, CoRL 2021 |
| `mandlekar2021robomimic` | Mandlekar *et al.*, robomimic, CoRL 2021 |
| `ross2011dagger` | Ross, Gordon, Bagnell, DAgger, AISTATS 2011 |
| `gupta2019relay` | Gupta *et al.*, Relay Policy Learning, CoRL 2019 |
| `ze2024dp3` | Ze *et al.*, 3D Diffusion Policy, arXiv:2403.03954 |

Metadatos extraidos del cuaderno *Diffusion Policy Extendido* (11 fuentes; las dos copias
del PDF de Song y Ermon son el mismo trabajo, de ahi que haya 10 entradas).

## Pendiente antes de entregar

- `main.tex`: titulo definitivo, autor, director y fecha.
- `img/logo-tecnun.pdf`: sin ese fichero la portada dibuja un recuadro con
  `[logo del centro]`.
- `bib/referencias.bib`: dos campos marcados `% verificar` (paginas de DDPM en las actas
  de NeurIPS 2020 y volumen/paginas de PMLR de Implicit Behavioral Cloning); no constan en
  la primera pagina de los PDF.
- `bib/bibliografia.bib`: autor y ano de la guia de estudio (`[PENDIENTE: …]`).
- Sustituir todo el texto entre corchetes de `secciones/`.
- `secciones/primera-hoja.tex` reproduce la hoja de identificacion; si el centro exige la
  plantilla oficial de Word, hay que sustituirla.
- Las figuras seran en color pero legibles en blanco y negro; las tablas nunca como imagen.

## Trampa de LaTeX que ya costo un cuelgue

Una linea de tabla que empieza por `[` justo despues de `\midrule` (o de `\\`) se
interpreta como su argumento opcional de anchura y **cuelga la compilacion sin dar error**.
Si una celda debe empezar por corchete, se escribe `{[texto]}`.
