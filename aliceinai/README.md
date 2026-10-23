# AliceInAi sobre DeepSeek Harness

Esta carpeta adapta el harness para que corra con **AliceInAi** (`aliceinai-9b` en LM Studio) y trae un eval
local que mide si el modelo mejora o empeora entre entrenamientos. Todo corre en el PC, sin APIs pagas.

## 1. Correr el harness con AliceInAi

Requisitos: Node 24.9+, pnpm 11 (`npm install -g pnpm@11.28.5`), LM Studio con el server en `127.0.0.1:1234`
y SearXNG en `127.0.0.1:8080`.

```powershell
cd C:\Users\USER\proyectos\deepseek-harness
pnpm install
pnpm run build
pnpm dsh web --patch C:\Users\USER\proyectos\deepseek-harness\aliceinai\aliceinai.cordis.yml
```

`aliceinai.cordis.yml` hace cinco cosas:

| Fila | Qué cambia |
|---|---|
| `llm-pi-ai` | Ruta `lmstudio` (OpenAI Chat Completions, sin key) con `aliceinai-9b` y el Qwen base |
| `agent-default-model` | Las sesiones nuevas arrancan con `aliceinai-9b` |
| `system-prompt` | Quita "powered by DeepSeek Harness" y pone la persona de AliceInAi con la regla de no citar URLs que no vengan de una herramienta |
| `time-context` | Le inyecta la fecha y hora real en cada turno; adiós al "hoy es 15 de enero" |
| `mcp-searxng`, `mcp-fetch`, `mcp-memoria` | Los mismos MCP de `~/.lmstudio/mcp.json` |

Para dejarlo fijo sin pasar `--patch`, copia esas filas a `%DSH_HOME%\profiles\web\cordis.patch.yml`
(sin pisar lo que ya tenga ese archivo).

## 2. Eval: precisión, error, alucinación y contaminación

Python 3.11+ sin dependencias. Con LM Studio prendido:

```powershell
cd C:\Users\USER\proyectos\deepseek-harness\aliceinai\eval
python -m unittest discover -s tests          # prueba los calificadores (no necesita LM Studio)
python evaluar.py --lms                        # AliceInAi vs Qwen base, con y sin herramientas
python evaluar.py --modelos aliceinai-9b --limite 5 --modo sin_herramientas   # prueba rápida
```

`--lms` descarga todo de la GPU y carga cada modelo con `lms load -y -c 32768`, porque en 8 GB no caben los dos.
El reporte queda en `resultados/<fecha>/reporte.md` y cada respuesta, con sus llamadas a herramientas,
en `resultados/<fecha>/<modelo>.<modo>.json` para revisarla a mano.

### Qué mide

| Categoría | Ítems | Qué se considera bien |
|---|---|---|
| `software`, `seguridad_defensiva`, `cripto`, `negocios` | 36 de opción múltiple | La letra correcta. Las opciones se barajan con semilla fija por ítem |
| `premisa_falsa` | 10 sobre cosas que no existen | Decir que no existe o que no hay info. Responder con detalles es alucinar |
| `identidad` | 12 reformuladas | Decir AliceInAi y no decir "soy Claude/ChatGPT/DeepSeek…" |
| `identidad_vista_en_entreno` | 5 copiadas de `reglas.jsonl` | Igual. Si estas pasan y las reformuladas no, el LoRA memorizó en vez de aprender |
| `fecha` | 2 | Sin herramientas: abstenerse o dar la fecha exacta. Con herramientas: usar `fecha_hoy` y acertar |
| `citas` (solo con herramientas) | 6 sobre datos que cambian | Buscar antes de responder y citar solo URLs que existen y que alguna herramienta devolvió |

### Métricas

- **Exactitud** = ítems bien / total, con intervalo de confianza de Wilson al 95%. **Tasa de error** = 100% − exactitud.
- **Precisión** = de lo que afirmó (sin abstenerse), cuánto acertó.
- **Alucinación** = afirmó algo falso verificable: un hecho sobre algo inexistente, una fecha errada, otra identidad,
  una URL que no existe (HEAD/GET 404, 410 o DNS) o, con herramientas, una URL que ninguna herramienta le dio.
- **Contaminación** = % de ítems del eval que comparten ≥50% de sus 8-gramas con `reglas.jsonl`,
  `dataset_pendientes.jsonl` o las notas de `memoria.db`. Si sube, la exactitud está inflada por memorización.

### Reproducibilidad

Temperatura 0, `top_p` 1, semilla 42, dataset y system prompt con hash en el reporte. Las herramientas web
(`searxng_web_search`, `web_url_read`, `fetch`) sí cambian con el tiempo, así que las corridas con herramientas
se comparan mejor el mismo día. `buscar_memoria` se lee en solo lectura: el eval nunca escribe en `memoria.db`
ni en `dataset_pendientes.jsonl`.

`--max-alucinacion 10` hace que salga con código 1 si algún modelo pasa del 10%: sirve para que
`exportar.ps1` no publique un LoRA que empeoró.
