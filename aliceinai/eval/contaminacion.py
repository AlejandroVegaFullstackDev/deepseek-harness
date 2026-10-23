"""Contaminación = preguntas del eval que se colaron en los datos de entrenamiento.

Si un ítem del eval aparece (casi) igual en reglas.jsonl o en las notas de memoria.db, el modelo
puede "acertar" por memorizar y el % de precisión queda inflado. Se mide con solapamiento de n-gramas
de palabras, la misma idea del chequeo de 13-gramas de GPT-3 pero con n=8 porque las preguntas son cortas.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from calificar import normalizar

N = 8


def _ngramas(texto: str, n: int = N) -> set[tuple[str, ...]]:
    palabras = [p for p in "".join(c if c.isalnum() else " " for c in normalizar(texto)).split()]
    if len(palabras) < n:
        return {tuple(palabras)} if palabras else set()
    return {tuple(palabras[i:i + n]) for i in range(len(palabras) - n + 1)}


def _textos_json(valor) -> list[str]:
    if isinstance(valor, str):
        return [valor]
    if isinstance(valor, dict):
        return [t for v in valor.values() for t in _textos_json(v)]
    if isinstance(valor, list):
        return [t for v in valor for t in _textos_json(v)]
    return []


def cargar_textos_entrenamiento(rutas: list[Path]) -> list[str]:
    textos: list[str] = []
    for ruta in rutas:
        if ruta.suffix == ".db":
            with sqlite3.connect(f"file:{ruta}?mode=ro", uri=True) as con:
                tablas = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
                if "notas" in tablas:
                    textos += [" ".join(str(c or "") for c in fila)
                               for fila in con.execute("select tema, contenido, fuente from notas")]
            continue
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            if linea.strip():
                textos.append(" ".join(_textos_json(json.loads(linea))))
    return textos


def indice(textos: list[str]) -> set[tuple[str, ...]]:
    total: set[tuple[str, ...]] = set()
    for t in textos:
        total |= _ngramas(t)
    return total


def solapamiento(texto_item: str, idx: set[tuple[str, ...]]) -> float:
    grams = _ngramas(texto_item)
    return len(grams & idx) / len(grams) if grams else 0.0
