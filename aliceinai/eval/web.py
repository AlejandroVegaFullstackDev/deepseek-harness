"""Acceso HTTP sin dependencias: cliente de LM Studio, herramientas web y verificación de URLs."""

from __future__ import annotations

import html
import json
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

AGENTE = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) aliceinai-eval/1.0"


def _peticion(url: str, *, metodo: str = "GET", cuerpo: dict | None = None, timeout: float = 30) -> tuple[int, bytes]:
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(url, data=datos, method=metodo, headers={
        "User-Agent": AGENTE,
        "Accept": "*/*",
        **({"Content-Type": "application/json"} if datos else {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if e.fp else b""


class LMStudio:
    def __init__(self, base_url: str, timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def modelos(self) -> list[str]:
        estado, cuerpo = _peticion(f"{self.base_url}/models", timeout=10)
        if estado != 200:
            raise RuntimeError(f"LM Studio no responde en {self.base_url} (HTTP {estado})")
        return [m["id"] for m in json.loads(cuerpo)["data"]]

    def chat(self, modelo: str, mensajes: list[dict], *, herramientas: list[dict] | None,
             semilla: int, max_tokens: int) -> dict:
        cuerpo = {
            "model": modelo,
            "messages": mensajes,
            "temperature": 0,
            "top_p": 1,
            "seed": semilla,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if herramientas:
            cuerpo["tools"] = herramientas
        estado, resp = _peticion(f"{self.base_url}/chat/completions", metodo="POST", cuerpo=cuerpo, timeout=self.timeout)
        if estado != 200:
            raise RuntimeError(f"HTTP {estado} de LM Studio: {resp[:300]!r}")
        return json.loads(resp)["choices"][0]["message"]


# Mismos nombres que exponen los MCP de Alejandro en LM Studio (mcp-searxng, mcp-server-fetch y memoria).
HERRAMIENTAS = [
    {"type": "function", "function": {
        "name": "searxng_web_search",
        "description": "Busca en la web con SearXNG. Devuelve título, URL y resumen de cada resultado.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "num_results": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "web_url_read",
        "description": "Lee el texto de una URL.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "fetch",
        "description": "Descarga una URL y devuelve su contenido como texto.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "buscar_memoria",
        "description": "Busca en lo que ya aprendiste antes. Usalo ANTES de buscar en la web.",
        "parameters": {"type": "object", "properties": {
            "consulta": {"type": "string"}, "k": {"type": "integer"}}, "required": ["consulta"]}}},
    {"type": "function", "function": {
        "name": "fecha_hoy",
        "description": "Devuelve la fecha de hoy.",
        "parameters": {"type": "object", "properties": {}}}},
]


class Herramientas:
    """Ejecuta las herramientas y anota cada URL que de verdad vio el modelo, para saber si sus citas tienen respaldo."""

    def __init__(self, searxng_url: str, hoy: date, memoria_db: Path | None = None):
        self.searxng_url = searxng_url.rstrip("/")
        self.hoy = hoy
        self.memoria_db = memoria_db
        self.urls_vistas: set[str] = set()
        self.llamadas: list[dict] = []

    def ejecutar(self, nombre: str, argumentos: dict) -> str:
        self.llamadas.append({"nombre": nombre, "argumentos": argumentos})
        try:
            if nombre == "searxng_web_search":
                return self._buscar(str(argumentos.get("query", "")), int(argumentos.get("num_results") or 5))
            if nombre in ("web_url_read", "fetch"):
                return self._leer(str(argumentos.get("url", "")))
            if nombre == "fecha_hoy":
                dias = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
                return f"Hoy es {dias[self.hoy.weekday()]} {self.hoy:%Y-%m-%d} (hora local)."
            if nombre == "buscar_memoria":
                return self._memoria(str(argumentos.get("consulta", "")), int(argumentos.get("k") or 5))
            return f"Herramienta '{nombre}' no disponible en el eval."
        except Exception as e:  # el modelo debe ver el error como lo vería en LM Studio
            return f"Error: {e}"

    def _buscar(self, consulta: str, n: int) -> str:
        q = urllib.parse.urlencode({"q": consulta, "format": "json"})
        estado, cuerpo = _peticion(f"{self.searxng_url}/search?{q}", timeout=30)
        if estado != 200:
            return f"Error: SearXNG HTTP {estado}"
        resultados = json.loads(cuerpo).get("results", [])[: max(1, min(n, 10))]
        lineas = []
        for r in resultados:
            self.urls_vistas.add(normalizar_url(r.get("url", "")))
            lineas.append(f"Title: {r.get('title', '')}\nDescription: {r.get('content', '')}\nURL: {r.get('url', '')}")
        return "\n\n".join(lineas) or "Sin resultados."

    def _memoria(self, consulta: str, k: int) -> str:
        """Igual que el FTS de memoria_mcp.py, pero en solo lectura y sin embeddings (así no toca la GPU)."""
        if not self.memoria_db or not self.memoria_db.exists():
            return "No hay nada guardado sobre eso. Busca en la web y luego usa guardar_conocimiento."
        terminos = " OR ".join(f'"{t}"' for t in consulta.replace('"', " ").split() if len(t) > 2)
        if not terminos:
            return "No hay nada guardado sobre eso."
        with sqlite3.connect(f"file:{self.memoria_db}?mode=ro", uri=True) as con:
            filas = con.execute(
                "SELECT n.id, n.tema, n.contenido, n.fuente FROM notas_fts f JOIN notas n ON n.id=f.rowid "
                "WHERE notas_fts MATCH ? ORDER BY rank LIMIT ?", (terminos, k)).fetchall()
        for _, _, _, fuente in filas:
            if fuente:
                self.urls_vistas.add(normalizar_url(fuente))
        if not filas:
            return "No hay nada guardado sobre eso. Busca en la web y luego usa guardar_conocimiento."
        return "\n\n".join(f"[{i}] {t}\n{c}\nFuente: {f or 'n/a'}" for i, t, c, f in filas)

    def _leer(self, url: str) -> str:
        estado, cuerpo = _peticion(url, timeout=30)
        if estado >= 400:
            return f"Error: HTTP {estado}"
        self.urls_vistas.add(normalizar_url(url))
        texto = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", cuerpo.decode("utf-8", "replace"))
        texto = html.unescape(re.sub(r"<[^>]+>", " ", texto))
        return re.sub(r"\s+", " ", texto).strip()[:6000]


def normalizar_url(url: str) -> str:
    p = urllib.parse.urlsplit(url.strip())
    host = p.netloc.lower().removeprefix("www.")
    return f"{host}{p.path.rstrip('/')}"


def verificar_url(url: str, cache: dict[str, str]) -> str:
    """'existe', 'no_existe' (404/410/DNS) o 'dudosa' (bloqueo anti-bot, timeout, 5xx)."""
    if url in cache:
        return cache[url]
    try:
        estado, _ = _peticion(url, metodo="HEAD", timeout=15)
        if estado in (403, 405, 429) or estado >= 500:
            estado, _ = _peticion(url, timeout=20)
        resultado = "existe" if estado < 400 else "no_existe" if estado in (404, 410) else "dudosa"
    except urllib.error.URLError as e:
        dns = "getaddrinfo" in str(e.reason) or "Name or service" in str(e.reason) or "11001" in str(e.reason)
        resultado = "no_existe" if dns else "dudosa"
    except Exception:
        resultado = "dudosa"
    cache[url] = resultado
    return resultado
