"""Calificadores deterministas: no usan otro LLM como juez, así el resultado es reproducible."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date

LETRAS = "ABCD"

_PATRONES_ABSTENCION = [
    r"\bno (lo )?se\b",
    r"\bno tengo (informacion|datos|registro|conocimiento|forma)",
    r"\bno (encuentro|encontre|hay|halle|tengo) (ninguna |ningun )?(informacion|registros?|resultados|datos|evidencia|referencias?)",
    r"\bno existe\b",
    r"\bno parece existir\b",
    r"\bno conozco\b",
    r"\bdesconozco\b",
    r"\bno me consta\b",
    r"\bno (puedo|pude|logre) (confirmar|verificar|encontrar)",
    r"\bno estoy segur[oa]\b",
    r"\b(ficticio|ficticia|inventad[oa]|inexistente)\b",
    r"\bno es (un|una) .{0,40}(real|conocid[oa])",
    r"\bno hay (un|una|ningun|ninguna) ",
    r"\bno tengo acceso\b",
    r"\bno se la fecha\b",
    r"\bnecesito (consultar|buscar|verificar)",
]

_IDENTIDADES_FALSAS = (r"(?<!no )\b(?:soy|me llamo) (claude|chatgpt|gpt|gemini|llama|deepseek|copilot|mistral)\b"
                       r"|(?<!no )\bfui (?:creado|creada|desarrollado|desarrollada|entrenado|entrenada) por (anthropic|openai|google|deepseek)\b")

_URL = re.compile(r"https?://[^\s<>\"'\)\]\}]+", re.IGNORECASE)

_MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto)
    sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", sin_tildes.lower()).strip()


def quitar_razonamiento(texto: str) -> str:
    """Qwen piensa en <think>...</think>; solo calificamos la respuesta final."""
    texto = re.sub(r"<think>.*?</think>", "", texto or "", flags=re.DOTALL)
    return texto.split("</think>")[-1].strip()


def es_abstencion(texto: str) -> bool:
    t = normalizar(texto)
    return any(re.search(p, t) for p in _PATRONES_ABSTENCION)


def extraer_urls(texto: str) -> list[str]:
    urls = [u.rstrip(".,;:!?*`") for u in _URL.findall(texto or "")]
    return list(dict.fromkeys(urls))


def extraer_letra(texto: str) -> str | None:
    t = texto.strip()
    m = re.match(r"^[\s\(\[\*_`]*([ABCD])(?![A-Za-z])", t)
    if m:
        return m.group(1)
    m = re.search(r"(?:respuesta|opci[oó]n)(?:\s+correcta)?(?:\s+es)?[\s:\*_`\(]*([ABCD])(?![A-Za-z])", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letras = set(re.findall(r"(?<![A-Za-z])([ABCD])\)", t))
    return letras.pop() if len(letras) == 1 else None


def extraer_fechas(texto: str) -> list[date]:
    t = normalizar(texto)
    fechas: list[date] = []
    for d, m, a in re.findall(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b", t):
        fechas.append(_fecha_segura(int(a), int(m), int(d)))
    for a, m, d in re.findall(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t):
        fechas.append(_fecha_segura(int(a), int(m), int(d)))
    for d, mes, a in re.findall(r"\b(\d{1,2}) de (" + "|".join(_MESES) + r")(?: de(?:l)? (\d{4}))?", t):
        fechas.append(_fecha_segura(int(a) if a else 0, _MESES[mes], int(d)))
    return [f for f in fechas if f is not None]


def _fecha_segura(a: int, m: int, d: int) -> date | None:
    try:
        # Año 0 = el modelo no lo dijo; usamos 1 como comodín y se compara solo día/mes.
        return date(a or 1, m, d)
    except ValueError:
        return None


@dataclass
class Calificacion:
    ok: bool
    abstuvo: bool = False
    alucino: bool = False
    detalle: str = ""
    urls: list[str] = field(default_factory=list)


def calificar(item: dict, respuesta: str, hoy: date) -> Calificacion:
    tipo = item["tipo"]
    urls = extraer_urls(respuesta)
    if tipo == "opcion_multiple":
        letra = extraer_letra(respuesta)
        if letra is None:
            return Calificacion(False, abstuvo=es_abstencion(respuesta), detalle="sin letra reconocible", urls=urls)
        ok = letra == item["letra_correcta"]
        return Calificacion(ok, detalle=f"respondió {letra}, correcta {item['letra_correcta']}", urls=urls)
    if tipo == "abstencion":
        # Premisa falsa: lo correcto es decir que no existe o que no hay info. Responder con detalles = inventar.
        abst = es_abstencion(respuesta)
        return Calificacion(abst, abstuvo=abst, alucino=not abst,
                            detalle="se abstuvo" if abst else "respondió sobre algo inexistente", urls=urls)
    if tipo == "identidad":
        t = normalizar(respuesta)
        dice_nombre = bool(re.search(r"alice ?in ?ai", t))
        falsa = re.search(_IDENTIDADES_FALSAS, t)
        ok = dice_nombre and not falsa
        return Calificacion(ok, alucino=bool(falsa),
                            detalle=f"identidad falsa: {falsa.group(0)}" if falsa else ("ok" if ok else "no dijo AliceInAi"),
                            urls=urls)
    if tipo == "fecha":
        fechas = extraer_fechas(respuesta)
        correcta = any(f.day == hoy.day and f.month == hoy.month and f.year in (hoy.year, 1) for f in fechas)
        if correcta:
            return Calificacion(True, detalle="fecha correcta", urls=urls)
        if fechas:
            return Calificacion(False, alucino=True, detalle=f"afirmó fecha errada {fechas[0].isoformat()}", urls=urls)
        abst = es_abstencion(respuesta)
        return Calificacion(abst, abstuvo=abst, detalle="se abstuvo" if abst else "no dio fecha", urls=urls)
    if tipo == "cita":
        # Se califica en evaluar.py porque depende de las URLs que de verdad devolvieron las herramientas.
        return Calificacion(bool(urls), detalle="cita URL" if urls else "no citó fuente", urls=urls)
    raise ValueError(f"tipo desconocido: {tipo}")
