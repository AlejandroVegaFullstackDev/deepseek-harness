"""Eval local y reproducible de AliceInAi contra su Qwen base, servidos por LM Studio.

Uso típico (desde esta carpeta, con LM Studio prendido en :1234 y SearXNG en :8080):

    python evaluar.py
    python evaluar.py --modelos aliceinai-9b --modo sin_herramientas --limite 5   # prueba rápida

Todo es stdlib: no hay nada que instalar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import contaminacion
from calificar import LETRAS, Calificacion, calificar, quitar_razonamiento
from web import HERRAMIENTAS, Herramientas, LMStudio, normalizar_url, verificar_url

AQUI = Path(__file__).resolve().parent
QWEN_APRENDE = Path(r"C:\Users\USER\proyectos\qwen-aprende")
MODOS = ("sin_herramientas", "con_herramientas")
MAX_RONDAS_HERRAMIENTAS = 6


def cargar_items(carpeta: Path) -> tuple[list[dict], str]:
    items, h = [], hashlib.sha256()
    for archivo in sorted(carpeta.glob("*.jsonl")):
        contenido = archivo.read_bytes()
        h.update(archivo.name.encode() + contenido)
        for linea in contenido.decode("utf-8").splitlines():
            if linea.strip():
                items.append(preparar(json.loads(linea)))
    return items, h.hexdigest()[:16]


def preparar(item: dict) -> dict:
    """Baraja las opciones con una semilla fija por id: misma letra en cada corrida, sin sesgo hacia la A."""
    if item["tipo"] != "opcion_multiple":
        item["prompt"] = item["pregunta"]
        return item
    opciones = list(item["opciones"])
    correcta = opciones[item["correcta"]]
    random.Random(item["id"]).shuffle(opciones)
    item["letra_correcta"] = LETRAS[opciones.index(correcta)]
    lineas = "\n".join(f"{LETRAS[i]}) {o}" for i, o in enumerate(opciones))
    item["prompt"] = f"{item['pregunta']}\n\n{lineas}\n\nResponde solo con la letra de la opción correcta."
    return item


def preguntar(cliente: LMStudio, modelo: str, sistema: str | None, item: dict, herramientas: Herramientas | None,
              semilla: int, max_tokens: int) -> tuple[str, list[dict]]:
    mensajes = ([{"role": "system", "content": sistema}] if sistema else []) + [{"role": "user", "content": item["prompt"]}]
    for _ in range(MAX_RONDAS_HERRAMIENTAS):
        msg = cliente.chat(modelo, mensajes, herramientas=HERRAMIENTAS if herramientas else None,
                           semilla=semilla, max_tokens=max_tokens)
        llamadas = msg.get("tool_calls") or []
        if not (herramientas and llamadas):
            return quitar_razonamiento(msg.get("content") or ""), herramientas.llamadas if herramientas else []
        mensajes.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": llamadas})
        for ll in llamadas:
            try:
                args = json.loads(ll["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            mensajes.append({"role": "tool", "tool_call_id": ll.get("id", ""),
                             "content": herramientas.ejecutar(ll["function"]["name"], args)})
    return "", herramientas.llamadas  # se quedó en bucle de herramientas sin responder


def revisar_urls(cal: Calificacion, herramientas: Herramientas | None, cache: dict[str, str]) -> dict:
    estados = {u: verificar_url(u, cache) for u in cal.urls}
    respaldadas = [u for u in cal.urls if herramientas and normalizar_url(u) in herramientas.urls_vistas]
    if any(e == "no_existe" for e in estados.values()):
        cal.ok, cal.alucino = False, True
        cal.detalle += " | cita URL que no existe"
    if herramientas and len(respaldadas) < len(cal.urls):
        cal.alucino = True
        cal.ok = False
        cal.detalle += " | cita URL que ninguna herramienta devolvió"
    return {"estados": estados, "respaldadas": respaldadas}


def calificar_con_modo(item: dict, respuesta: str, modo: str, hoy: date, llamadas: list[dict]) -> Calificacion:
    cal = calificar(item, respuesta, hoy)
    if modo == "con_herramientas" and item["tipo"] == "fecha" and cal.abstuvo:
        cal.ok, cal.detalle = False, "tenía fecha_hoy y no la usó"
    if item["tipo"] == "cita" and not any(ll["nombre"] == "searxng_web_search" for ll in llamadas):
        cal.ok, cal.detalle = False, "no buscó antes de responder"
    return cal


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centro - margen), min(1.0, centro + margen)


def metricas(filas: list[dict]) -> dict:
    n = len(filas)
    ok = sum(f["ok"] for f in filas)
    abst = sum(f["abstuvo"] for f in filas)
    aluc = sum(f["alucino"] for f in filas)
    afirmadas = [f for f in filas if not f["abstuvo"]]
    estados = [e for f in filas for e in f["urls"]["estados"].values()]
    con_url = sum(len(f["urls"]["estados"]) for f in filas)
    return {
        "n": n,
        "exactitud": ok / n if n else 0,
        "exactitud_ic95": wilson(ok, n),
        "tasa_error": 1 - ok / n if n else 0,
        "precision": sum(f["ok"] for f in afirmadas) / len(afirmadas) if afirmadas else 0,
        "tasa_alucinacion": aluc / n if n else 0,
        "alucinacion_ic95": wilson(aluc, n),
        "tasa_abstencion": abst / n if n else 0,
        "urls_citadas": con_url,
        "urls_existen": estados.count("existe"),
        "urls_no_existen": estados.count("no_existe"),
        "urls_dudosas": estados.count("dudosa"),
        "urls_respaldadas": sum(len(f["urls"]["respaldadas"]) for f in filas),
    }


def lms(*args: str) -> None:
    if not shutil.which("lms"):
        print("  (no encontré el CLI lms; LM Studio cargará el modelo por JIT)")
        return
    subprocess.run(["lms", *args], check=False)


def correr(args: argparse.Namespace) -> int:
    cliente = LMStudio(args.base_url)
    disponibles = cliente.modelos()
    faltan = [m for m in args.modelos if m not in disponibles]
    if faltan:
        print(f"Estos modelos no están en LM Studio: {faltan}. Disponibles: {disponibles}")
        return 2

    items, hash_items = cargar_items(args.datos)
    if args.limite:
        items = items[: args.limite]
    sistema_txt = args.system_prompt.read_text(encoding="utf-8") if args.system_prompt else None
    hoy = args.fecha

    rutas_entreno = [p for p in args.entrenamiento if p.exists()]
    idx = contaminacion.indice(contaminacion.cargar_textos_entrenamiento(rutas_entreno))
    contaminados = {}
    for it in items:
        if it.get("vista_en_entreno"):  # están a propósito: miden memorización, no cuentan como fuga
            continue
        texto = it["pregunta"] + " " + str(it.get("opciones", [""])[it["correcta"]] if "correcta" in it else "")
        s = contaminacion.solapamiento(texto, idx)
        if s >= args.umbral_contaminacion:
            contaminados[it["id"]] = round(s, 3)

    sello = datetime.now().strftime("%Y%m%d-%H%M%S")
    salida = args.salida / sello
    salida.mkdir(parents=True, exist_ok=True)
    modos = MODOS if args.modo == "ambos" else (args.modo,)
    cache_urls: dict[str, str] = {}
    resumen: dict = {
        "fecha_corrida": sello, "fecha_referencia": hoy.isoformat(), "base_url": args.base_url,
        "semilla": args.semilla, "max_tokens": args.max_tokens, "items": len(items), "hash_items": hash_items,
        "hash_system_prompt": hashlib.sha256(sistema_txt.encode()).hexdigest()[:16] if sistema_txt else None,
        "datos_entrenamiento": [str(p) for p in rutas_entreno],
        "contaminacion": {"tasa": len(contaminados) / max(1, sum(not i.get("vista_en_entreno") for i in items)), "items": contaminados,
                          "umbral": args.umbral_contaminacion, "ngramas": contaminacion.N},
        "resultados": {},
    }

    for modelo in args.modelos:
        if args.lms:
            print(f"Cargando {modelo} con lms (descargo los demás para liberar la GPU)...")
            lms("unload", "--all")
            lms("load", modelo, "-y", "-c", "32768")
        for modo in modos:
            filas = []
            print(f"\n== {modelo} · {modo} ==")
            for i, it in enumerate(items, 1):
                if it["tipo"] == "cita" and modo == "sin_herramientas":
                    continue
                herr = Herramientas(args.searxng, hoy, args.memoria_db) if modo == "con_herramientas" else None
                t0 = time.time()
                try:
                    respuesta, llamadas = preguntar(cliente, modelo, sistema_txt, it, herr, args.semilla, args.max_tokens)
                except Exception as e:
                    respuesta, llamadas = f"[ERROR] {e}", herr.llamadas if herr else []
                cal = calificar_con_modo(it, respuesta, modo, hoy, llamadas)
                urls = revisar_urls(cal, herr, cache_urls)
                filas.append({
                    "id": it["id"], "categoria": it["categoria"], "tipo": it["tipo"], "ok": cal.ok,
                    "abstuvo": cal.abstuvo, "alucino": cal.alucino, "detalle": cal.detalle, "urls": urls,
                    "herramientas": llamadas, "segundos": round(time.time() - t0, 1),
                    "pregunta": it["prompt"], "respuesta": respuesta,
                })
                marca = "OK " if cal.ok else ("ALU" if cal.alucino else "MAL")
                print(f"  [{i:>3}/{len(items)}] {marca} {it['id']:<28} {cal.detalle[:70]}")
            por_cat = defaultdict(list)
            for f in filas:
                por_cat[f["categoria"]].append(f)
            resumen["resultados"].setdefault(modelo, {})[modo] = {
                "global": metricas(filas),
                "por_categoria": {c: metricas(fs) for c, fs in sorted(por_cat.items())},
            }
            (salida / f"{modelo}.{modo}.json").write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")

    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    (salida / "reporte.md").write_text(reporte(resumen), encoding="utf-8")
    print(f"\nListo. Reporte: {salida / 'reporte.md'}")

    peor = max((r["global"]["tasa_alucinacion"] for m in resumen["resultados"].values() for r in m.values()), default=0)
    if args.max_alucinacion is not None and peor * 100 > args.max_alucinacion:
        print(f"FALLA: alucinación {peor:.1%} supera el máximo de {args.max_alucinacion}%")
        return 1
    return 0


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def reporte(r: dict) -> str:
    c = r["contaminacion"]
    out = [
        "# Eval AliceInAi",
        "",
        f"Corrida {r['fecha_corrida']} · fecha de referencia {r['fecha_referencia']} · {r['items']} ítems "
        f"(hash {r['hash_items']}) · temperatura 0 · semilla {r['semilla']} · system prompt {r['hash_system_prompt']}",
        "",
        "## Resumen",
        "",
        "| Modelo | Modo | Exactitud (IC 95%) | Tasa de error | Precisión | Alucinación (IC 95%) | Abstención | URLs citadas | URLs inexistentes | URLs sin respaldo |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for modelo, modos in r["resultados"].items():
        for modo, res in modos.items():
            g = res["global"]
            lo, hi = g["exactitud_ic95"]
            alo, ahi = g["alucinacion_ic95"]
            sin_respaldo = g["urls_citadas"] - g["urls_respaldadas"] if modo == "con_herramientas" else "n/a"
            out.append(
                f"| {modelo} | {modo} | {pct(g['exactitud'])} ({pct(lo)}–{pct(hi)}) | {pct(g['tasa_error'])} | "
                f"{pct(g['precision'])} | {pct(g['tasa_alucinacion'])} ({pct(alo)}–{pct(ahi)}) | {pct(g['tasa_abstencion'])} | "
                f"{g['urls_citadas']} | {g['urls_no_existen']} | {sin_respaldo} |")
    out += ["", "## Por categoría", ""]
    for modelo, modos in r["resultados"].items():
        for modo, res in modos.items():
            out += [f"### {modelo} · {modo}", "", "| Categoría | n | Exactitud | Alucinación |", "|---|---|---|---|"]
            out += [f"| {cat} | {m['n']} | {pct(m['exactitud'])} | {pct(m['tasa_alucinacion'])} |"
                    for cat, m in res["por_categoria"].items()]
            out.append("")
    out += [
        "## Contaminación",
        "",
        f"{pct(c['tasa'])} de los ítems del eval comparten ≥{int(c['umbral'] * 100)}% de sus {c['ngramas']}-gramas con "
        f"los datos de entrenamiento ({', '.join(r['datos_entrenamiento']) or 'ninguno encontrado'}).",
    ]
    if c["items"]:
        out += ["", "Ítems contaminados (sácalos del entrenamiento o del eval):", ""]
        out += [f"- `{k}`: {pct(v)}" for k, v in c["items"].items()]
    out += [
        "",
        "## Cómo leerlo",
        "",
        "- **Exactitud**: ítems bien resueltos sobre el total. **Tasa de error** = 100% − exactitud.",
        "- **Precisión**: de las veces que afirmó algo (sin abstenerse), cuántas acertó.",
        "- **Alucinación**: afirmó algo falso verificable: un hecho sobre algo que no existe, una fecha errada, "
        "otra identidad, una URL que no existe, o (con herramientas) una URL que ninguna búsqueda le devolvió.",
        "- **Abstención**: dijo que no sabe o que no existe. En preguntas con premisa falsa es lo correcto.",
        "- Con pocos ítems el intervalo de confianza es ancho: compara modelos solo si los intervalos no se pisan.",
    ]
    return "\n".join(out) + "\n"


def principal() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelos", nargs="+", default=["aliceinai-9b", "qwen3.8-9b-distill-uncensored-heretic"])
    p.add_argument("--modo", choices=(*MODOS, "ambos"), default="ambos")
    p.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    p.add_argument("--searxng", default="http://127.0.0.1:8080")
    p.add_argument("--system-prompt", type=Path,
                   default=QWEN_APRENDE / "system_prompt.txt" if (QWEN_APRENDE / "system_prompt.txt").exists() else None)
    p.add_argument("--sin-system-prompt", action="store_true", help="evalúa el modelo pelado, sin system prompt")
    p.add_argument("--entrenamiento", type=Path, nargs="*", default=[
        QWEN_APRENDE / "datos" / "reglas.jsonl",
        QWEN_APRENDE / "datos" / "dataset_pendientes.jsonl",
        QWEN_APRENDE / "datos" / "memoria.db",
    ])
    p.add_argument("--memoria-db", type=Path, default=QWEN_APRENDE / "datos" / "memoria.db",
                   help="memoria.db para buscar_memoria (se abre solo lectura; el eval nunca escribe ahí)")
    p.add_argument("--datos", type=Path, default=AQUI / "datos")
    p.add_argument("--salida", type=Path, default=AQUI / "resultados")
    p.add_argument("--fecha", type=date.fromisoformat, default=date.today(), help="fecha 'de hoy' esperada (AAAA-MM-DD)")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--limite", type=int, default=0, help="solo los primeros N ítems")
    p.add_argument("--umbral-contaminacion", type=float, default=0.5)
    p.add_argument("--max-alucinacion", type=float, default=None, help="sale con código 1 si algún modelo supera este %%")
    p.add_argument("--lms", action="store_true", help="usa el CLI lms para descargar todo y cargar cada modelo")
    args = p.parse_args()
    if args.sin_system_prompt:
        args.system_prompt = None
    return correr(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(principal())
