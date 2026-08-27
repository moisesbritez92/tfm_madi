#!/usr/bin/env bash
# Compila la memoria con una unica bibliografia IEEE:
#   main.aux -> Referencias (bib/referencias.bib)
set -e
cd "$(dirname "$0")"
DOC=main
pdflatex -interaction=nonstopmode -halt-on-error "$DOC" >/dev/null
bibtex "$DOC"        || true
pdflatex -interaction=nonstopmode -halt-on-error "$DOC" >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error "$DOC" >/dev/null
echo "Listo: $DOC.pdf"
grep -c "Warning" "$DOC.log" | xargs -I{} echo "avisos en el log: {}"
