#!/usr/bin/env python3
"""Revisa un .tex de beamer y avisa de lo que suele delatar una charla floja.

Uso:
    python3 check_deck.py main.tex [--minutes 15] [--lang es|en]

No sustituye a mirar el PDF: comprueba lo mecánico (densidad, temas por defecto,
tablas mal puestas, muletillas) para que la revisión humana se centre en el fondo.
Devuelve 1 si hay errores, 0 si solo hay avisos.
"""
import argparse
import re
import sys
from pathlib import Path

RESET, RED, YEL, GRN, DIM = "\033[0m", "\033[31m", "\033[33m", "\033[32m", "\033[2m"

BAD_THEMES = ("Warsaw Madrid Berlin Berkeley CambridgeUS Antibes Copenhagen Frankfurt "
              "Singapore Bergen Boadilla Dresden Darmstadt Goettingen Hannover Ilmenau "
              "JuanLesPins Luebeck Malmoe Marburg Montpellier PaloAlto Pittsburgh "
              "Rochester Szeged Warsaw").split()

FILLER = {
    "es": [r"en el mundo actual", r"cabe (?:señalar|destacar|mencionar)",
           r"es importante (?:destacar|señalar|mencionar|notar)",
           r"en resumen,? hemos", r"a lo largo de (?:este|la) (?:trabajo|presentación)",
           r"profundiza(?:r|mos) en", r"aborda(?:r|mos)? la problemática",
           r"enfoque (?:holístico|novedoso|innovador)", r"de última generación",
           r"revolucionari[oa]", r"robusto y escalable", r"gracias por su atención",
           r"¿?preguntas\?", r"gran importancia", r"juega un papel (?:fundamental|clave)"],
    "en": [r"in today'?s world", r"it is important to note", r"delve into",
           r"cutting[- ]edge", r"state[- ]of[- ]the[- ]art approach", r"leverag(?:e|ing)",
           r"a testament to", r"in conclusion,? we have presented", r"paradigm shift",
           r"thank you for your attention", r"any questions\?", r"seamless(?:ly)?",
           r"game[- ]chang", r"holistic approach"],
}

GENERIC_TITLES = {"introducción", "introduccion", "introduction", "metodología",
                  "metodologia", "methodology", "methods", "resultados", "results",
                  "conclusiones", "conclusion", "conclusions", "motivación", "motivacion",
                  "motivation", "trabajo futuro", "future work", "discusión", "discussion",
                  "estado del arte", "related work", "background", "objetivos"}

DECOR = re.compile(r"[\U0001F300-\U0001FAFF\u2700-\u27BF\u2B00-\u2BFF\u2190-\u21FF"
                   r"\u2600-\u26FF\u2714\u2716\u2605\u2606]")


class Report:
    def __init__(self):
        self.items = []

    def add(self, level, where, msg, hint=""):
        self.items.append((level, where, msg, hint))

    def show(self):
        order = {"ERROR": 0, "AVISO": 1, "NOTA": 2}
        colors = {"ERROR": RED, "AVISO": YEL, "NOTA": DIM}
        for lvl, where, msg, hint in sorted(self.items, key=lambda x: order[x[0]]):
            loc = f" [{where}]" if where else ""
            print(f"{colors[lvl]}{lvl}{RESET}{loc} {msg}")
            if hint:
                print(f"      {DIM}→ {hint}{RESET}")
        n_err = sum(1 for i in self.items if i[0] == "ERROR")
        n_warn = sum(1 for i in self.items if i[0] == "AVISO")
        if not self.items:
            print(f"{GRN}Sin incidencias mecánicas.{RESET}")
        print(f"\n{len(self.items)} incidencias: {n_err} errores, {n_warn} avisos.")
        return n_err


def strip_comments(tex):
    return re.sub(r"(?<!\\)%.*", "", tex)


def get_frames(tex):
    """Devuelve [(titulo, cuerpo, nº de línea)] de cada frame."""
    frames = []
    for m in re.finditer(r"\\begin\{frame\}(.*?)\\end\{frame\}", tex, re.S):
        body = m.group(1)
        line = tex[:m.start()].count("\n") + 1
        title = ""
        t = re.match(r"\s*(?:\[[^\]]*\])?\s*\{(.*?)\}", body, re.S)
        if t:
            title = " ".join(t.group(1).split())
        else:
            ft = re.search(r"\\frametitle\{(.*?)\}", body, re.S)
            if ft:
                title = " ".join(ft.group(1).split())
        frames.append((title, body, line))
    return frames


def visible_words(body):
    """Palabras que el público lee de verdad: sin matemáticas, sin comandos, sin código."""
    s = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]|\$[^$]*\$", " ", body, flags=re.S)
    s = re.sub(r"\\begin\{(equation|align|gather|tikzpicture|axis|tabular|lstlisting|"
               r"verbatim)\*?\}.*?\\end\{\1\*?\}", " ", s, flags=re.S)
    s = re.sub(r"\\(?:label|ref|cite\w*|includegraphics|input|include|usepackage|"
               r"pgfplotsset|addplot|src)\s*(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"[{}\[\]&\\~^_]", " ", s)
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ'-]+", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--minutes", type=int, default=None,
                    help="duración prevista de la charla, para contrastar el número de diapositivas")
    ap.add_argument("--lang", default="both", choices=["es", "en", "both"])
    ap.add_argument("--max-words", type=int, default=45)
    args = ap.parse_args()

    path = Path(args.tex)
    raw = path.read_text(encoding="utf-8", errors="replace")
    tex = strip_comments(raw)
    r = Report()

    # El estilo puede vivir en un .sty o en un fichero incluido: para las comprobaciones
    # de preámbulo miramos también esos ficheros, si están al lado.
    extra = ""
    for sty in path.parent.glob("*.sty"):
        extra += strip_comments(sty.read_text(encoding="utf-8", errors="replace"))
    for m in re.finditer(r"\\(?:input|include)\{([^}]*)\}", tex):
        cand = path.parent / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
        if cand.exists():
            extra += strip_comments(cand.read_text(encoding="utf-8", errors="replace"))
    preamble = tex + extra

    # ---------- preámbulo / estilo
    for theme in BAD_THEMES:
        if re.search(r"\\usetheme\s*(\[[^\]]*\])?\{" + theme + r"\}", tex):
            r.add("ERROR", "preámbulo", f"tema por defecto '{theme}'",
                  "es el aspecto de fábrica de beamer; usa talkstyle.sty o metropolis")
    if "navigation symbols" not in preamble and not re.search(r"\\usetheme\{metropolis\}", preamble):
        r.add("AVISO", "preámbulo", "los símbolos de navegación no están desactivados",
              r"añade \setbeamertemplate{navigation symbols}{}")
    if re.search(r"\\begin\{tabular\}", tex) and "booktabs" not in preamble:
        r.add("AVISO", "preámbulo", "hay tablas pero no se carga booktabs",
              "las líneas de \\hline dan aspecto de hoja de cálculo")
    if re.search(r"\\appendix", tex) and "appendixnumberbeamer" not in preamble:
        r.add("NOTA", "preámbulo", "hay apéndice sin appendixnumberbeamer",
              "las diapositivas de respaldo inflan el contador n/N")
    if not re.search(r"aspectratio=169", tex):
        r.add("NOTA", "preámbulo", "no se ve aspectratio=169",
              "casi todos los proyectores actuales son 16:9")

    # ---------- tablas y figuras
    for m in re.finditer(r"\\begin\{tabular\}\s*(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        if "|" in m.group(1):
            r.add("AVISO", f"línea {tex[:m.start()].count(chr(10)) + 1}",
                  "tabla con líneas verticales", "quítalas y usa \\toprule/\\midrule/\\bottomrule")
    if re.search(r"\\hline", tex):
        r.add("AVISO", "tablas", "uso de \\hline",
              "sustitúyelo por las reglas de booktabs")
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        if m.group(1).lower().endswith((".png", ".jpg", ".jpeg")):
            r.add("NOTA", "figuras", f"imagen de mapa de bits: {m.group(1)}",
                  "en proyección se ve pixelada; regenera en PDF si es posible")
    if re.search(r"\\resizebox[^\n]*\\(?:\[|begin\{(?:equation|align))", tex) or \
       re.search(r"\\(?:tiny|scriptsize)\s*\\begin\{(?:equation|align)", tex):
        r.add("AVISO", "matemáticas", "hay ecuaciones encogidas para que quepan",
              "es señal de diapositiva sobrecargada: pártela en dos")

    # ---------- por frame
    frames = get_frames(tex)
    if not frames:
        r.add("ERROR", "", "no se ha encontrado ningún \\begin{frame}")
        return r.show()

    generic = 0
    for title, body, line in frames:
        where = f"«{title[:40]}»" if title else f"línea {line}"
        words = visible_words(body)
        if len(words) > args.max_words:
            r.add("AVISO", where, f"{len(words)} palabras visibles",
                  f"por encima de ~{args.max_words} el público lee en vez de escuchar")
        items = len(re.findall(r"\\item\b", body))
        if items > 6:
            r.add("AVISO", where, f"{items} viñetas",
                  "más de 5–6 no se retienen; parte la diapositiva")
        displays = len(re.findall(r"\\\[|\\begin\{(?:equation|align|gather|multline)\*?\}", body))
        if displays > 2:
            r.add("AVISO", where, f"{displays} ecuaciones en display",
                  "una por diapositiva; dos solo si la segunda se deriva de la primera")
        if len(re.findall(r"\\textbf\{", body)) > 3:
            r.add("NOTA", where, "demasiadas negritas",
                  "si todo destaca, nada destaca")
        if len(set(re.findall(r"\\textcolor\{([^}]*)\}", body))) > 2:
            r.add("NOTA", where, "más de dos colores distintos en la misma diapositiva",
                  "un solo color de acento dirige la mirada")
        if DECOR.search(body) or DECOR.search(title):
            r.add("ERROR", where, "emoji o símbolo decorativo",
                  "no encaja en un registro académico")
        if re.search(r"\\begin\{(?:verbatim|lstlisting|minted)\}", body):
            head = body[:body.find("{")] if "{" in body else body[:40]
            if "fragile" not in head:
                r.add("ERROR", where, "código verbatim en un frame sin [fragile]",
                      r"usa \begin{frame}[fragile]{...} o no compilará")
        if title.strip().lower().rstrip(":.") in GENERIC_TITLES:
            generic += 1
            r.add("NOTA", where, "título genérico",
                  "un título que afirme algo: «El error cae un 23 %», no «Resultados»")

    if generic >= 3:
        r.add("AVISO", "guion", f"{generic} títulos genéricos de manual",
              "el mazo se lee como un informe; convierte los títulos en mensajes")

    # ---------- muletillas
    langs = ["es", "en"] if args.lang == "both" else [args.lang]
    low = tex.lower()
    for lg in langs:
        for pat in FILLER[lg]:
            for m in re.finditer(pat, low):
                frag = low[m.start():m.end()][:45]
                r.add("AVISO", f"línea {low[:m.start()].count(chr(10)) + 1}",
                      f"muletilla de relleno: «{frag}»",
                      "dilo con la boca o bórralo; en la diapositiva no aporta")

    # ---------- ritmo
    body_frames = [f for f in frames if "noframenumbering" not in f[1][:80]]
    n = len(body_frames)
    print(f"{DIM}{len(frames)} frames en total, {n} numerados.{RESET}\n")
    if args.minutes:
        lo, hi = int(args.minutes * 0.7), int(args.minutes * 1.3)
        if n > hi:
            r.add("AVISO", "ritmo", f"{n} diapositivas para {args.minutes} min",
                  f"para ese tiempo caben unas {lo}–{hi}; manda material al apéndice")
        elif n < lo:
            r.add("NOTA", "ritmo", f"solo {n} diapositivas para {args.minutes} min",
                  "puede que falte desarrollo, o que cada diapositiva esté muy cargada")
    if "\\appendix" not in tex:
        r.add("NOTA", "ritmo", "no hay diapositivas de respaldo",
              "3–10 de apéndice salvan el turno de preguntas")

    return r.show()


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
