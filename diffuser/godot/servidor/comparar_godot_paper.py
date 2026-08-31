#!/usr/bin/env python3
"""Contraste preregistrado entre V0 y el punto de control publicado, dentro de Godot.

El protocolo esta fijado por adelantado en ``memoria/preregistro_godot_paper.md``.
Este guion no decide nada: ejecuta lo que ese documento declara.

Ocho celdas, que son dos brazos por dos condiciones de observacion por dos
realizaciones de ruido, cincuenta condiciones iniciales cada una:

  brazos          v0, v_paper
  condiciones     estado (A, solo cambia el motor de fisica)
                  godot  (B, cambia el motor y los pixeles) -- la primaria
  realizaciones   prueba  (semilla base 20260827)
                  ruido_b (semilla base 20260831)
  semillas        200000-200049, no condicionadas al exito de nadie

Tres cosas que lo separan de ``barrido_color.py``, del que hereda la forma:

  * **Levanta y mata el servidor el mismo.** ``--variante``, ``--obs`` y
    ``--base-seed`` son argumentos de arranque del servidor, asi que cada celda
    necesita su propia sesion. Nunca hay dos a la vez: son 8 GB de VRAM y dos
    contextos CUDA en esa tarjeta ya mataron un entrenamiento en julio.
  * **No parsea la salida por pantalla.** Le pasa ``salida=`` a Godot y lee el
    JSON escrito. Es obligatorio, no cosmetico: el nombre por defecto de
    ``main.gd`` no lleva la realizacion, y la segunda tanda machacaria la primera
    sin decir nada.
  * **Agrega en el formato de ``logs_entrenamiento/prueba_final/``**, para que
    ``memoria/scripts/analisis_godot_paper.py`` pueda reutilizar el aparato de
    analisis que ya existe en lugar de copiarlo.

Es reanudable por celda y por episodio: trece horas no caben en una sesion.

Uso (Windows, desde la raiz del repositorio):
    .venv_diffuser_infer\\Scripts\\python.exe diffuser/godot/servidor/comparar_godot_paper.py --portones
    .venv_diffuser_infer\\Scripts\\python.exe diffuser/godot/servidor/comparar_godot_paper.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
GODOT_DIR = AQUI.parent
DIFFUSER = GODOT_DIR.parent
RAIZ = DIFFUSER.parent
for ruta in (str(DIFFUSER / "scripts"), str(AQUI)):
    if ruta not in sys.path:
        sys.path.insert(0, ruta)

from barrido_color import resolver_godot  # noqa: E402
from evaluar_bloque_test import (  # noqa: E402  (fija HF_HUB_OFFLINE al importar)
    SEMILLAS_DEMOSTRACIONES,
    SEMILLAS_SELECCION,
    sha256,
)
from evaluar_paper_bloque_test import SEMILLAS_SELECCION_PAPER  # noqa: E402

PRUEBA_FINAL = RAIZ / "logs_entrenamiento" / "prueba_final"
SALIDA = RAIZ / "logs_entrenamiento" / "godot_paper"
EPISODIOS = GODOT_DIR / "grabaciones" / "godot_paper"

# Bloque preregistrado. Prefijo del bloque final 200000-200199.
SEMILLA_INICIAL = 200000
N_SEMILLAS = 50

# Las dos realizaciones de ruido, con las semillas base del preregistro hermano.
REALIZACIONES = {"prueba": 20260827, "ruido_b": 20260831}

# Condicion de observacion -> bandera del servidor. La B es la primaria.
CONDICIONES = {"a": "estado", "b": "godot"}

BRAZOS = {
    "v0": ("V0", "ResNet-18 desde cero"),
    "v_paper": ("V_PAPER", "ResNet-18 con spatial softmax de robomimic"),
}

# Puntos de control congelados en el preregistro, por SHA-256.
SHA_ESPERADO = {
    "v0": "5310551ee71075d9efcf956c34670809741d84e06808809551e7675674e8ce63",
    "v_paper": "bac7221f7e34cd51162dc1972e1a39ffcddc87de1dc1780c44ffa61b88c4ff76",
}

# Porton 2. Referencias y tolerancia, fijadas en el preregistro.
SEMILLAS_DERIVA = list(range(200000, 200008))
REFERENCIA_DERIVA = {"v0": PRUEBA_FINAL / "deriva_v0.json",
                     "v_paper": PRUEBA_FINAL / "prueba_paper.json"}
TOLERANCIA_DERIVA = 0.07

PUERTO = 5580
SEGUNDOS_ARRANQUE = 240
SEGUNDOS_EPISODIO = 400
UMBRAL_EXITO = 0.999


# --- utilidades ---------------------------------------------------------------


def commit() -> str:
    try:
        salida = subprocess.run(
            ["git", "-C", str(RAIZ), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return salida.stdout.strip()
    except Exception:  # descriptivo, no carga con nada
        return "desconocido"


def _nvidia_smi(campo: str) -> str:
    try:
        salida = subprocess.run(
            ["nvidia-smi", f"--query-gpu={campo}", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return salida.stdout.strip().splitlines()[0]
    except Exception:  # descriptivo, no carga con nada
        return "desconocido"


def entorno() -> dict:
    """Version de la pila y tarjeta, **sin inicializar CUDA en este proceso**.

    `torch.cuda.get_device_name` haria un `_lazy_init` y este proceso se quedaria
    con un contexto CUDA vivo durante las trece horas, restandoselo a cada uno de
    los ocho servidores que va a levantar. `torch.__version__` y
    `torch.version.cuda` no tocan el dispositivo; el nombre de la tarjeta sale del
    controlador.
    """
    import torch

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": _nvidia_smi("name"),
        "controlador": _nvidia_smi("driver_version"),
    }


def punto_control(brazo: str) -> Path:
    """Ruta del punto de control congelado, por el modulo de inferencia del brazo."""
    import importlib

    if str(DIFFUSER) not in sys.path:
        sys.path.insert(0, str(DIFFUSER))
    from servidor_politica import MODELOS

    return importlib.import_module(MODELOS[brazo]).default_checkpoint()


def semillas(desde: int, cuantas: int) -> list:
    return list(range(desde, desde + cuantas))


def nombre_celda(realizacion: str, brazo: str, condicion: str) -> Path:
    return SALIDA / f"{realizacion}_{brazo}_{condicion}.json"


def nombre_episodio(realizacion: str, brazo: str, condicion: str, seed: int) -> Path:
    return EPISODIOS / f"{realizacion}_{brazo}_{condicion}_seed{seed}.json"


# --- porton 1: protocolo, sin GPU ---------------------------------------------


def porton_protocolo(args) -> bool:
    """SHA-256 de los dos puntos de control, disyuncion del bloque y utileria."""
    print("Porton 1: protocolo")
    ok = True

    bloque = set(semillas(args.semillas_desde, args.cuantas))
    for nombre, otro in (
        ("las demostraciones", set(SEMILLAS_DEMOSTRACIONES)),
        ("el conjunto de seleccion de V0", set(SEMILLAS_SELECCION)),
        ("el conjunto de seleccion del articulo", set(SEMILLAS_SELECCION_PAPER)),
    ):
        cruce = bloque & otro
        marca = "ok" if not cruce else f"TOCA {sorted(cruce)[:5]}"
        ok &= not cruce
        print(f"  bloque disjunto de {nombre}: {marca}")

    for brazo in args.brazos:
        ruta = punto_control(brazo)
        if not ruta.is_file():
            print(f"  {brazo}: FALTA {ruta}")
            ok = False
            continue
        print(f"  {brazo}: calculando SHA-256 de {ruta.name} ...", end="", flush=True)
        digest = sha256(ruta)
        bien = digest == SHA_ESPERADO[brazo]
        ok &= bien
        print(f"\r  {brazo}: {ruta.name} -> "
              f"{'ok' if bien else 'NO ES EL PUNTO DE CONTROL DEL PREREGISTRO'}")

    try:
        import robomimic  # noqa: F401
        print("  robomimic importable: ok")
    except Exception as error:
        print(f"  robomimic importable: NO ({type(error).__name__}: {error})")
        ok = False

    try:
        print(f"  Godot: {resolver_godot(args.godot)}")
    except SystemExit as error:
        print(f"  Godot: NO ({error})")
        ok = False

    print(f"  -> porton 1 {'PASA' if ok else 'NO PASA'}\n")
    return ok


# --- porton 2: deriva de entorno ----------------------------------------------


def porton_deriva(args) -> bool:
    """Cada brazo contra el pymunk original, desde el servidor de Windows.

    Lo que acota este porton es el salto de `robodiff` en WSL (torch 1.12.1,
    CUDA 11.6) a `.venv_diffuser_infer` en Windows (torch 2.6.0, CUDA 12.4), con
    la fisica fija en el simulador original: si aqui no hay deriva grande,
    cualquier caida posterior es atribuible a Godot y no al cambio de entorno.

    No acota solo la version de torch, y conviene no exagerar lo que dice. El
    esquema de siembra del servidor indexa por semilla de episodio y el del
    evaluador de WSL indexa por tanda, de modo que **la corriente de ruido no es
    la misma** y una coincidencia bit a bit es imposible por construccion. De ahi
    que el criterio sea una tolerancia sobre la media y no una igualdad.
    """
    from cliente_prueba import Cliente, comprobar_estado_inicial, episodio

    print("Porton 2: deriva de entorno (pymunk original, servidor de Windows)")
    ok = True
    for brazo in args.brazos:
        referencia = json.loads(
            REFERENCIA_DERIVA[brazo].read_text(encoding="utf-8")
        )["puntuaciones"]
        esperada = sum(referencia[str(s)] for s in SEMILLAS_DERIVA) / len(SEMILLAS_DERIVA)

        proceso = arrancar_servidor(brazo, "estado", REALIZACIONES["prueba"], args)
        try:
            cliente = Cliente("127.0.0.1", args.puerto)
            try:
                info = cliente.pedir({"cmd": "hola"})
                print(f"  {info['variante']} | {info['punto_control']} | "
                      f"{info['dispositivo']}")
                estados_ok = comprobar_estado_inicial(cliente, SEMILLAS_DERIVA[:4])
                filas = [episodio(cliente, s) for s in SEMILLAS_DERIVA]
            finally:
                cliente.cerrar()
        finally:
            parar_servidor(proceso)

        media = sum(f["max_reward"] for f in filas) / len(filas)
        desvio = abs(media - esperada)
        pasa = desvio <= TOLERANCIA_DERIVA and estados_ok
        ok &= pasa
        print(f"  {brazo}: media {media:.4f} frente a {esperada:.4f} de "
              f"{REFERENCIA_DERIVA[brazo].name} | desvio {desvio:.4f} "
              f"(tolerancia {TOLERANCIA_DERIVA}) | estados iniciales "
              f"{'ok' if estados_ok else 'DIFIEREN'} -> {'PASA' if pasa else 'NO PASA'}",
              flush=True)

    if not ok:
        print("  Contingencia declarada en el preregistro: el contraste se ejecuta\n"
              "  igualmente, pero el secundario 3 (deriva contra prueba_final) no se\n"
              "  reporta.")
    print(f"  -> porton 2 {'PASA' if ok else 'NO PASA'}\n")
    return ok


# --- servidor -----------------------------------------------------------------


def saludar(puerto: int, timeout: float = 5.0) -> dict | None:
    """Un `hola` completo por el protocolo. None si no hay nadie que conteste.

    Una conexion TCP a secas **no** vale como prueba de que el servidor esta
    listo, y costo un porton descubrirlo: la sonda del arranque de V_PAPER
    conecto con el zocalo del servidor anterior, que todavia se estaba cerrando,
    y dio por bueno un servidor que aun estaba cargando pesos. El cliente de
    verdad llego despues y se encontro la conexion rechazada. Un `hola` que hay
    que responder no se puede confundir con eso.
    """
    try:
        with socket.create_connection(("127.0.0.1", puerto), timeout=timeout) as sock:
            sock.settimeout(timeout)
            canal = sock.makefile("rwb")
            canal.write(json.dumps({"cmd": "hola"}).encode("utf-8") + b"\n")
            canal.flush()
            linea = canal.readline()
            return json.loads(linea) if linea.strip() else None
    except (OSError, ValueError):
        return None


def arrancar_servidor(brazo: str, obs: str, base_seed: int, args) -> subprocess.Popen:
    """Levanta el servidor y espera a que **conteste** siendo quien debe ser.

    La espera no termina hasta que el saludo confirma el brazo y el modo de
    observacion pedidos. En un contraste pareado, servir la celda con el punto de
    control equivocado seria el error mas caro posible y el mas silencioso: las
    cifras saldrian, y saldrian mal.
    """
    esperar_puerto_libre(args.puerto)
    orden = [
        sys.executable, str(AQUI / "servidor_politica.py"),
        "--variante", brazo, "--obs", obs,
        "--puerto", str(args.puerto), "--base-seed", str(base_seed),
    ]
    proceso = subprocess.Popen(orden)
    limite = time.time() + SEGUNDOS_ARRANQUE
    while time.time() < limite:
        if proceso.poll() is not None:
            raise SystemExit(f"el servidor murio al arrancar (codigo {proceso.returncode})")
        saludo = saludar(args.puerto)
        if saludo and saludo.get("ok"):
            variante = str(saludo.get("variante", "")).lower()
            if variante == brazo and saludo.get("modo_obs") == obs:
                return proceso
            parar_servidor(proceso)
            raise SystemExit(
                f"el servidor del puerto {args.puerto} dice ser {variante!r} en "
                f"obs={saludo.get('modo_obs')!r}, se pidio {brazo!r} en obs={obs!r}"
            )
        time.sleep(1.0)
    parar_servidor(proceso)
    raise SystemExit(f"el servidor no contesto en el puerto {args.puerto} en "
                     f"{SEGUNDOS_ARRANQUE} s")


def esperar_puerto_libre(puerto: int, segundos: int = 60) -> None:
    """No arranca uno nuevo mientras el anterior siga escuchando."""
    limite = time.time() + segundos
    while time.time() < limite:
        try:
            with socket.create_connection(("127.0.0.1", puerto), timeout=1):
                time.sleep(1.0)
        except OSError:
            return
    raise SystemExit(f"el puerto {puerto} sigue ocupado despues de {segundos} s")


def parar_servidor(proceso: subprocess.Popen) -> None:
    """Lo mata y **confirma** que ha muerto antes de devolver el control.

    Sin la confirmacion, la celda siguiente arranca su servidor mientras el
    anterior todavia tiene su contexto CUDA vivo, y en 8 GB eso no cabe.
    """
    if proceso.poll() is not None:
        return
    proceso.terminate()
    try:
        proceso.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proceso.kill()
        proceso.wait(timeout=30)


# --- episodios ----------------------------------------------------------------


def correr_episodio(godot: str, brazo: str, condicion: str, realizacion: str,
                    seed: int, args) -> dict | None:
    """Un episodio de Godot en modo grabar. Devuelve el resumen que escribio."""
    destino = nombre_episodio(realizacion, brazo, condicion, seed)
    if destino.is_file() and not args.forzar:
        return json.loads(destino.read_text(encoding="utf-8"))

    obs = CONDICIONES[condicion]
    relativa = f"res://grabaciones/godot_paper/{destino.name}"
    orden = [godot]
    # La condicion B necesita renderizador: sin ventana no hay SubViewport que leer.
    if obs == "estado":
        orden.append("--headless")
    orden += [
        "--path", str(GODOT_DIR), "--",
        "modo=grabar", f"obs={obs}", f"seed={seed}",
        f"puerto={args.puerto}", "perturbacion=ninguna",
        f"variante={brazo}", f"salida={relativa}",
    ]
    try:
        subprocess.run(orden, capture_output=True, text=True,
                       timeout=args.tiempo)
    except subprocess.TimeoutExpired:
        return None
    if not destino.is_file():
        return None
    return json.loads(destino.read_text(encoding="utf-8"))


def correr_celda(godot: str, realizacion: str, brazo: str, condicion: str,
                 args) -> None:
    destino = nombre_celda(realizacion, brazo, condicion)
    if destino.is_file() and not args.forzar:
        print(f"{destino.name} ya existe, se salta (--forzar para rehacerla)")
        return

    lista = semillas(args.semillas_desde, args.cuantas)
    base = REALIZACIONES[realizacion]
    etiqueta, codificador = BRAZOS[brazo]
    ruta_ckpt = punto_control(brazo)

    print(f"\n=== {etiqueta} | condicion {condicion.upper()} ({CONDICIONES[condicion]}) "
          f"| realizacion {realizacion} (base {base}) | {len(lista)} semillas ===",
          flush=True)
    proceso = arrancar_servidor(brazo, CONDICIONES[condicion], base, args)
    inicio = time.time()
    puntuaciones, reintentadas, perdidas = {}, [], []
    try:
        for i, seed in enumerate(lista, 1):
            resumen = correr_episodio(godot, brazo, condicion, realizacion, seed, args)
            if resumen is None:
                # El preregistro permite un reintento y solo uno.
                reintentadas.append(seed)
                resumen = correr_episodio(godot, brazo, condicion, realizacion,
                                          seed, args)
            if resumen is None:
                perdidas.append(seed)
                print(f"  [{i:2d}/{len(lista)}] {seed}  PERDIDA", flush=True)
                continue
            puntuaciones[seed] = float(resumen["recompensa_max"])
            print(f"  [{i:2d}/{len(lista)}] {seed}  {puntuaciones[seed]:.4f}  "
                  f"{resumen['pasos']:3d} pasos  "
                  f"({(time.time() - inicio) / 60:.0f} min)", flush=True)
    finally:
        parar_servidor(proceso)

    media = sum(puntuaciones.values()) / len(puntuaciones) if puntuaciones else float("nan")
    exitos = sum(1 for v in puntuaciones.values() if v >= UMBRAL_EXITO)
    resultado = {
        "variante": etiqueta,
        "codificador": codificador,
        "brazo": brazo,
        "condicion": condicion,
        "obs": CONDICIONES[condicion],
        "realizacion": realizacion,
        "simulador": "Godot 4 (port de Push-T)",
        "perturbacion": "ninguna",
        "punto_control": ruta_ckpt.name,
        "sha256": SHA_ESPERADO[brazo],
        "n_test": len(lista),
        "test_start_seed": args.semillas_desde,
        "base_seed_difusion": base,
        "max_pasos": 300,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "media": media,
        "exitos": exitos,
        "segundos": round(time.time() - inicio, 1),
        "episodios_reintentados": reintentadas,
        "episodios_perdidos": perdidas,
        "entorno": ".venv_diffuser_infer (Windows)",
        **entorno(),
        "precision": "float32",
        "fecha_iso": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": commit(),
        "puntuaciones": {str(s): puntuaciones[s] for s in lista if s in puntuaciones},
    }
    SALIDA.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print(f"  media {media:.4f} | exito {exitos}/{len(lista)} | "
          f"{(time.time() - inicio) / 60:.0f} min | escrito en {destino}", flush=True)
    if perdidas:
        print(f"  ATENCION: {len(perdidas)} celdas perdidas: {perdidas}")


# --- principal ----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brazos", nargs="+", default=list(BRAZOS), choices=list(BRAZOS))
    parser.add_argument("--condiciones", nargs="+", default=list(CONDICIONES),
                        choices=list(CONDICIONES),
                        help="a: solo cambia la fisica. b: la fisica y los pixeles")
    parser.add_argument("--realizaciones", nargs="+", default=list(REALIZACIONES),
                        choices=list(REALIZACIONES))
    parser.add_argument("--semillas-desde", type=int, default=SEMILLA_INICIAL)
    parser.add_argument("--cuantas", type=int, default=N_SEMILLAS)
    parser.add_argument("--puerto", type=int, default=PUERTO)
    parser.add_argument("--tiempo", type=int, default=SEGUNDOS_EPISODIO,
                        help="segundos por episodio antes de darlo por colgado")
    parser.add_argument("--godot", default="")
    parser.add_argument("--portones", action="store_true",
                        help="ejecuta los portones 1 y 2 y termina")
    parser.add_argument("--solo-comprobaciones", action="store_true",
                        help="solo el porton 1, que no necesita GPU")
    parser.add_argument("--forzar", action="store_true",
                        help="rehace celdas y episodios ya escritos")
    args = parser.parse_args()

    if not porton_protocolo(args):
        raise SystemExit("el porton 1 no pasa; no se ejecuta nada")
    if args.solo_comprobaciones:
        return 0

    if args.portones:
        porton_deriva(args)
        print("Porton 3, a mano y con ventana, antes de la pasada larga:\n"
              "  servidor_politica.py --variante v_paper --obs godot --puerto 5555\n"
              "  .\\lanzar.ps1 -Obs godot -Semilla 200000\n"
              "Criterio: el episodio termina y NO aparece el aviso de historial plano.")
        return 0

    if args.semillas_desde != SEMILLA_INICIAL or args.cuantas != N_SEMILLAS:
        raise SystemExit(
            "el bloque esta fijado en el preregistro (200000-200049) y el tamano de "
            "muestra no se mueve en funcion del resultado. Para otra cosa, edita el "
            "preregistro primero y deja constancia."
        )

    godot = resolver_godot(args.godot)
    EPISODIOS.mkdir(parents=True, exist_ok=True)
    inicio = time.time()
    # Primero todas las celdas de la condicion A, que corren sin ventana, y
    # despues las de la B, que la necesitan y roban el foco. Asi las seis horas
    # y media de A se pueden dejar corriendo mientras se usa el equipo, y solo
    # las de B piden la maquina para ellas.
    for condicion in args.condiciones:
        for realizacion in args.realizaciones:
            for brazo in args.brazos:
                correr_celda(godot, realizacion, brazo, condicion, args)
    print(f"\nTotal {(time.time() - inicio) / 3600:.1f} h. Analisis:\n"
          "  python memoria/scripts/analisis_godot_paper.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
