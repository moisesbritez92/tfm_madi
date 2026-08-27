"""Controles de integridad del zarr de Push-T (m6 del informe).

Se abre en modo lectura y se leen solo los arrays necesarios: `img` es float32 de
25650x96x96x3 y cargarlo entero cuesta 2,84 GB de RAM. Se recorre por trozos.
"""
import hashlib, json, os, sys
import numpy as np, zarr

RUTA = os.path.expanduser("~/tfm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr")
z = zarr.open(RUTA, "r")
data, meta = z["data"], z["meta"]
fin = np.asarray(meta["episode_ends"])
res = {"ruta": RUTA, "n_episodios": int(fin.size), "arrays": {}}

for nombre in sorted(data.array_keys()):
    a = data[nombre]
    info = {"forma": list(a.shape), "dtype": str(a.dtype), "chunks": list(a.chunks)}
    if nombre == "img":
        nan = inf = 0; mn, mx = np.inf, -np.inf
        for i in range(0, a.shape[0], 512):
            b = np.asarray(a[i:i+512])
            nan += int(np.isnan(b).sum()); inf += int(np.isinf(b).sum())
            mn = min(mn, float(b.min())); mx = max(mx, float(b.max()))
    else:
        b = np.asarray(a[:])
        nan = int(np.isnan(b).sum()); inf = int(np.isinf(b).sum())
        mn, mx = float(b.min()), float(b.max())
    info.update(nan=nan, inf=inf, min=round(mn, 6), max=round(mx, 6))
    res["arrays"][nombre] = info

n = int(fin[-1])
res["n_transiciones"] = n
res["episode_ends_creciente"] = bool(np.all(np.diff(fin) > 0))
res["longitud_min"] = int(np.diff(np.concatenate([[0], fin])).min())
res["longitud_max"] = int(np.diff(np.concatenate([[0], fin])).max())
res["longitudes_coherentes"] = all(
    data[k].shape[0] == n for k in data.array_keys()
)

# duplicados exactos de estado (fuga potencial entre episodios)
st = np.asarray(data["state"][:])
uniq = np.unique(st, axis=0)
res["estados_duplicados"] = int(st.shape[0] - uniq.shape[0])

# hash del fichero: zarr es un directorio, se resume su contenido ordenado
h = hashlib.sha256()
for raiz, dirs, ficheros in sorted(os.walk(RUTA)):
    dirs.sort()
    for f in sorted(ficheros):
        p = os.path.join(raiz, f)
        h.update(os.path.relpath(p, RUTA).encode())
        with open(p, "rb") as fh:
            for trozo in iter(lambda: fh.read(1 << 20), b""):
                h.update(trozo)
res["sha256_arbol"] = h.hexdigest()
res["bytes_en_disco"] = sum(
    os.path.getsize(os.path.join(r, f))
    for r, _, fs in os.walk(RUTA) for f in fs
)
json.dump(res, sys.stdout, indent=2, ensure_ascii=False)
print()
