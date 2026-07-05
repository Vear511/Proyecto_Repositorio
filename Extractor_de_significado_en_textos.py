from ollama import Client
import pandas as pd
import os
import json
import threading
import time                   
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
# ─────────────────────────────────────────────
#  CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────

# Modelo de Ollama que se usará para la extracción
OLLAMA_MODEL_ID = 'gpt-oss:120b'

# Cantidad de descripciones que se envían al modelo en cada petición
BATCH_SIZE = 300

# Hilos paralelos para procesar múltiples batches al mismo tiempo
NUM_THREADS = 3

# Ruta donde se persiste la configuración del usuario entre sesiones
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".clasificador_config.json")

# Dimensiones fijas para los campos de entrada de una sola línea.
# grid_propagate(False) en _uniform_frame hace que estos valores
# se respeten aunque el contenido interior sea más pequeño o más grande.
INPUT_H = 24   # altura en píxeles
INPUT_W = 420  # ancho en píxeles


# ── Centinelas como Enum ─────────────────────────────────────────────────────
# Usar un Enum en lugar de strings mágicos elimina el riesgo de colisión con
# cualquier descripción real que pudiera contener esos textos, y hace las
# comparaciones explícitas y seguras frente a futuros cambios de nombre.
class Centinela(Enum):
    FALLIDO   = '__FALLIDO__'    # el lote agotó los 3 reintentos devolviendo todo 'otro'
    PENDIENTE = '__PENDIENTE__'  # el lote fue cancelado antes de procesarse


# Prompt por defecto que instruye al LLM sobre cómo clasificar las descripciones.
# El usuario puede editarlo en la interfaz; se persiste entre sesiones.
DEFAULT_PROMPT = """Olvida toda tarea pasada y cumple la siguiente tarea.
Tarea: clasificar descripciones de activos fijos.

Recibirás una lista numerada de descripciones. Para cada una debes responder con su número seguido de dos puntos y la categoría, una por línea.

CATEGORÍAS PERMITIDAS (debes elegir EXCLUSIVAMENTE una de estas, tal cual están escritas):
- arriendo
- bodega
- cancha
- casa
- central hidroelectrica
- centro comercial
- construccion
- departamento
- edificio
- equipo
- equipo agricola
- equipo tecnologico
- equipo electrico
- herramienta
- equipo medico
- equipo industrial
- estacionamiento
- explotacion forestal
- galpones
- infraestructura
- inmueble
- instalacion equipo
- local comercial
- maquinaria
- vehiculo
- muebles
- oficinas
- planta
- plantacion
- terreno
- viviendas

Reglas estrictas:
-. Solo puedes responder con una categoría de la lista anterior.
-. Elige la categoría que mejor describa el activo fijo.
"""
PROMPT_OBLIGATORIO = """Reglas obligatorias
-. Si es ambigua, vacía o no encaja, responde 'otro'.
-. Debes responder EXACTAMENTE la misma cantidad de líneas que descripciones recibiste.
-. Respeta las tildes y caracteres especiales exactamente como están escritas.

Prohibido en la respuesta: Explicaciones, puntuación extra, comillas, mayúsculas.

Formato obligatorio:
1: [Respuesta 1]
2: [Respuesta 2]
3: [Respuesta 3]
"""

# Caché en memoria: evita volver a llamar al LLM para descripciones ya procesadas.
# Se limpia al inicio de cada extracción para garantizar resultados frescos.
cache_clasificaciones: dict = {}

# Lock que protege lecturas y escrituras concurrentes sobre la caché.
# Aunque en la práctica las escrituras ocurren en un único hilo (_proceso),
# el lock garantiza corrección si en el futuro se lanzan sesiones paralelas.
_cache_lock = threading.Lock()


# ─────────────────────────────────────────────
#  PERSISTENCIA DE CONFIGURACIÓN
# ─────────────────────────────────────────────

def cargar_config() -> dict:
    """
    Lee el archivo JSON de configuración del usuario.
    Devuelve un dict vacío si el archivo no existe o no es válido.
    Guarda: api_key y ultimo_prompt.
    """
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def guardar_config(data: dict) -> None:
    """
    Persiste el dict de configuración en disco.
    Falla silenciosamente para no interrumpir el flujo principal.
    """
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# ─────────────────────────────────────────────
#  UTILIDADES PURAS
# ─────────────────────────────────────────────

def normalizar(v) -> str | None:
    """
    Convierte un valor a string limpio, o None si es vacío/inválido.

    Definida a nivel de módulo (no anidada) para poder reutilizarla tanto en
    clasificar_con_cache como en extraer_batch sin duplicar la lógica de filtrado.
    """
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return None if (not s or s.lower() == 'nan' or s in ('[]', '.')) else s

def filtrar_por_columna(df: pd.DataFrame, columna: str, valor: str) -> pd.DataFrame:
    """
    Filtra el DataFrame dejando solo las filas donde `columna` es igual a `valor`.
    Si columna o valor no son validos, o valor es '(todos)', devuelve el
    DataFrame completo sin filtrar.
    """
    if not columna or columna not in df.columns or not valor or valor == '(todos)':
        return df
    return df[df[columna].astype(str) == str(valor)].reset_index(drop=True)

def filtrar_por_rango(df: pd.DataFrame, desde: int, hasta: int) -> pd.DataFrame:
    """
    Restringe el DataFrame a un rango especifico de filas (1-indexado, inclusive).
    Si los valores son invalidos o vacios, devuelve el DataFrame completo.
    """
    n = len(df)
    ini = max(1, desde) if desde else 1
    fin = min(n, hasta) if hasta else n
    if ini > fin:
        return df
    return df.iloc[ini-1:fin].reset_index(drop=True)

def generar_grafico_resultados(df: pd.DataFrame, columna_resultado: str, ruta_salida: str) -> str:
    """
    Genera un grafico de barras con el conteo de cada categoria en columna_resultado
    y lo guarda como imagen PNG en ruta_salida. Devuelve la ruta del archivo generado.
    """
    conteo = df[columna_resultado].value_counts()
    plt.figure(figsize=(10, 6))
    conteo.plot(kind="bar", color="#1a6496")
    plt.title("Distribucion de categorias extraidas")
    plt.xlabel("Categoria")
    plt.ylabel("Cantidad de filas")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    return ruta_salida

def cargar_dataframe(ruta: str) -> pd.DataFrame:
    """
    Carga un archivo tabular en un DataFrame de pandas.
    Soporta '.csv', '.tsv', '.txt'  y '.xlsx', '.xls', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'.
    """
    EXTENSIONES_EXCEL = {'.xlsx', '.xls', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'}
    EXTENSIONES_CSV   = {'.csv', '.tsv', '.txt'}

    # Se extrae la extension del archivo en minusculas para comparar con los sets
    ext = os.path.splitext(ruta)[1].lower()

    if ext in EXTENSIONES_EXCEL:
        return pd.read_excel(ruta)
    elif ext in EXTENSIONES_CSV:
        separador = '\t' if ext == '.tsv' else None
        return pd.read_csv(ruta, sep=separador, engine='python', encoding='utf-8-sig')
    else:
        return pd.read_csv(ruta, sep=None, engine='python', encoding='latin-1')


# ─────────────────────────────────────────────
#  LÓGICA DE EXTRACCIÓN
# ─────────────────────────────────────────────

def extraer_batch(
    comentarios: list,
    prompt_template: str,
    stop_event: threading.Event,
    client: Client,
    model_id: str,
) -> tuple[list, list, bool, str]:
    """
    Extrae información de un batch de descripciones usando el LLM con hasta MAX_REINTENTOS.

    Parada cooperativa:
        Antes de cada intento al LLM se comprueba stop_event. Si está activo, el batch
        se cancela inmediatamente y se devuelve batch_ok=False sin consumir más tokens.
        Nota: una llamada ya en vuelo a Ollama no puede interrumpirse desde Python;
        la parada aplica al intento SIGUIENTE o al siguiente batch.

    Un intento se considera fallido cuando el modelo devuelve 'otro' para TODAS las
    descripciones del batch, lo que suele indicar un problema de comprensión o saturación.
    En ese caso se reintenta hasta agotar los intentos disponibles.

    Parámetros:
        comentarios     : lista de strings del batch actual.
        prompt_template : texto del sistema que guía la extracción.
        stop_event      : evento compartido; si está activo se cancela el batch.
        client          : instancia de Client de Ollama creada en _proceso().
                          Recibirlo por parámetro desacopla la función del estado
                          global y la hace testeable de forma unitaria.
        model_id        : identificador del modelo a usar en la llamada al LLM.
                          Idem anterior.

    Retorna:
        resultados             : lista de categorías en el mismo orden que comentarios.
        logs                   : mensajes de depuración para mostrar en la UI.
        batch_ok               : False si el batch fue cancelado o falló tras MAX_REINTENTOS.
        ultima_respuesta_cruda : última respuesta de Ollama (para diagnóstico en la UI).
    """
    resultados:     list      = []
    items_para_llm: list[str] = []
    indices_llm:    list[int] = []
    logs:           list[str] = []

    # --- Filtrado previo usando normalizar() de módulo: omitir vacíos, nulos, inútiles ---
    for i, com in enumerate(comentarios):
        val = normalizar(com)
        if val is None:
            logs.append(f"[{i}] '{str(com)[:50]}' → vacío")
            resultados.append((i, ''))
        else:
            items_para_llm.append(val)
            indices_llm.append(i)

    # Si todas eran vacías, no hay nada que enviar al LLM
    if not items_para_llm:
        return [r for _, r in sorted(resultados)], logs, True, ""

    # --- Construcción del prompt de usuario con las descripciones numeradas ---
    bloque = "\n".join(f"{j+1}. {c}" for j, c in enumerate(items_para_llm))
    messages_input = [
        {"role": "system", "content": prompt_template},
        {"role": "user",   "content": f"Descripciones:\n{bloque}\n\nRespuesta:"}
    ]

    MAX_REINTENTOS         = 3
    ultima_respuesta_cruda = ""

    for intento in range(MAX_REINTENTOS):

        # ── Punto de parada cooperativa ──────────────────────────────────────
        # Se comprueba ANTES de cada llamada a Ollama para no iniciar trabajo
        # innecesario cuando el usuario (o un batch fallido previo) pidió detener.
        if stop_event.is_set():
            logs.append(f"⏹ Batch cancelado por stop_event antes del intento {intento + 1}.")
            for idx in indices_llm:
                resultados.append((idx, Centinela.PENDIENTE))
            return [r for _, r in sorted(resultados)], logs, False, ""
        # ─────────────────────────────────────────────────────────────────────

        try:
            response = client.chat(model=model_id, messages=messages_input, stream=False)
            ultima_respuesta_cruda = response['message']['content']

            # --- Parseo de la respuesta: "N: categoria" por línea ---
            categorias_llm: dict[int, str] = {}
            for linea in ultima_respuesta_cruda.strip().splitlines():
                if ':' in linea:
                    num_str, _, cat = linea.partition(':')
                    try:
                        categorias_llm[int(num_str.strip())] = cat.strip().lower() or 'otro'
                    except ValueError:
                        pass  # línea malformada; el índice recibirá 'otro' por defecto

            # --- Mapeo de resultados al orden original del batch ---
            resultados_llm = []
            for j, idx in enumerate(indices_llm):
                categoria = categorias_llm.get(j + 1, 'otro')
                resultados_llm.append((idx, categoria or 'otro'))

            categorias_obtenidas = [cat for _, cat in resultados_llm]

            # --- Detección de respuesta generada: todo 'otro' ---
            if all(cat == 'otro' for cat in categorias_obtenidas):
                if intento < MAX_REINTENTOS - 1:
                    logs.append(f"⚠ Intento {intento + 1}/{MAX_REINTENTOS}: todo 'otro', reintentando...")
                    continue
                else:
                    # Agotados todos los intentos — loguear respuesta cruda y marcar como fallido
                    logs.append(
                        f"❌ Lote sin extraer tras {MAX_REINTENTOS} intentos.\n"
                        f"Respuesta cruda de Ollama:\n"
                        f"{'─' * 40}\n"
                        f"{ultima_respuesta_cruda[:150]}"
                        f"{ '...' if len(ultima_respuesta_cruda) > 150 else '' }\n"
                        f"{'─' * 40}"
                    )
                    resultados.extend(resultados_llm)
                    return [r for _, r in sorted(resultados)], logs, False, ultima_respuesta_cruda
            else:
                # Respuesta válida — loguear cada ítem y devolver con éxito
                for j, (idx, categoria) in enumerate(resultados_llm):
                    logs.append(f"[{idx}] '{items_para_llm[j][:50]}' → {categoria}")
                resultados.extend(resultados_llm)
                return [r for _, r in sorted(resultados)], logs, True, ultima_respuesta_cruda

        except Exception as e:
            ultima_respuesta_cruda = f"Excepción: {e}"
            logs.append(f"Error en LLM (intento {intento + 1}/{MAX_REINTENTOS}): {e}")
            if intento < MAX_REINTENTOS - 1:
                continue
            else:
                # Error de red/timeout agotado — asignar 'otro' y marcar fallido
                logs.append(f"❌ Lote fallido tras {MAX_REINTENTOS} intentos. Último error: {e}")
                for idx in indices_llm:
                    resultados.append((idx, 'otro'))
                return [r for _, r in sorted(resultados)], logs, False, ultima_respuesta_cruda

    # Rama de seguridad: el bucle anterior siempre devuelve antes de llegar aquí.
    return [r for _, r in sorted(resultados)], logs, False, ultima_respuesta_cruda


def clasificar_con_cache(
    valores: list,
    prompt_template: str,
    log_fn,
    stop_event: threading.Event,
    client: Client,   # propagado desde _proceso() hacia extraer_batch
    model_id: str,    # propagado desde _proceso() hacia extraer_batch
    progress_callback=None,   # callable(completed:int, total:int)
    eta_callback=None,        # callable(segundos_restantes: float | None)
) -> tuple[list, bool, str]:
    """
    Punto de entrada principal para la extracción de una columna entera.

    Optimización por caché:
    - Sólo envía al LLM las descripciones únicas no vistas todavía.
    - Los valores ya extraídos se reutilizan directamente de cache_clasificaciones.

    Paralelismo con parada temprana:
    - Divide los valores únicos pendientes en batches de BATCH_SIZE elementos.
    - Procesa los batches en paralelo con ThreadPoolExecutor (NUM_THREADS hilos).
    - El mismo stop_event sirve para DOS propósitos:
        a) Parada por fallo: se activa cuando un batch no consigue resultado válido.
        b) Parada por usuario: se activa desde el botón "Detener" de la UI.
      En ambos casos los batches restantes se marcan con Centinela.PENDIENTE
      sin llamar al LLM.

    ETA (tiempo restante estimado):
    - Se mide el tiempo transcurrido desde que comenzó el procesamiento.
    - Tras cada lote completado se calcula: promedio = elapsed / completados.
      Luego: eta_segundos = (total - completados) * promedio.
    - Se comunica a la UI mediante eta_callback para actualizar el label en tiempo real.
    - Cuando no quedan lotes pendientes se llama con None para limpiar el label.

    Parámetros:
        valores         : lista de valores de la columna seleccionada.
        prompt_template : prompt del sistema para el LLM.
        log_fn          : función de logging thread-safe para la UI.
        stop_event      : evento compartido con la UI para paradas externas.
        client          : instancia de Client de Ollama; se pasa a extraer_batch.
        model_id        : identificador del modelo; se pasa a extraer_batch.
        progress_callback : callable(completed, total) para actualizar la barra.
        eta_callback    : callable(segundos_restantes | None) para actualizar el ETA.

    Retorna:
        resultados  : lista de categorías (puede incluir Centinela.FALLIDO/PENDIENTE).
        hubo_fallo  : True si se detuvo por un batch fallido (no por el usuario).
        razon_fallo : descripción legible del motivo de la parada por fallo.
    """
    valores_norm = [normalizar(v) for v in valores]

    # Únicos que aún no están en caché (lectura protegida por lock)
    with _cache_lock:
        unicos_pendientes = list({v for v in valores_norm
                                  if v and v not in cache_clasificaciones})

    log_fn(f"Filas totales: {len(valores)}   |   Únicos a extraer: {len(unicos_pendientes)}")

    # hubo_fallo=True solo cuando un batch agota sus reintentos (no cuando el usuario para)
    hubo_fallo  = False
    razon_fallo = ""

    if unicos_pendientes:
        batches = [unicos_pendientes[i:i + BATCH_SIZE]
                   for i in range(0, len(unicos_pendientes), BATCH_SIZE)]
        total       = len(batches)
        completados = 0
        if progress_callback:
            progress_callback(0, total)

        # ── Reloj de inicio para el cálculo de ETA ───────────────────────────
        # Se captura justo antes del primer submit, no antes de la validación,
        # para que el promedio refleje el tiempo real de llamadas a la API.
        tiempo_inicio = time.monotonic()
        # ─────────────────────────────────────────────────────────────────────

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            # stop_event, client y model_id se pasan a cada worker
            futures = {
                executor.submit(
                    extraer_batch, b, prompt_template, stop_event, client, model_id
                ): b
                for b in batches
            }

            for future in as_completed(futures):
                batch_original = futures[future]
                completados += 1

                if progress_callback:
                    progress_callback(completados, total)

                # ── Cálculo de ETA tras cada lote completado ─────────────────
                # Usamos tiempo de pared (wall-clock) dividido entre lotes
                # completados para obtener el throughput real incluyendo
                # paralelismo y reintentos.
                elapsed = time.monotonic() - tiempo_inicio
                if completados > 0 and completados < total:
                    avg_por_lote = elapsed / completados
                    segundos_restantes = (total - completados) * avg_por_lote
                    if eta_callback:
                        eta_callback(segundos_restantes)
                elif completados >= total:
                    # Todos los lotes terminaron → limpiar el label
                    if eta_callback:
                        eta_callback(None)
                # ─────────────────────────────────────────────────────────────

                # Si el evento ya estaba activo (fallo previo O usuario), cancelar sin procesar
                if stop_event.is_set():
                    log_fn(f"\n── Lote {completados}/{total}: CANCELADO (extracción detenida) ──")
                    with _cache_lock:
                        for descripcion in batch_original:
                            if descripcion not in cache_clasificaciones:
                                cache_clasificaciones[descripcion] = Centinela.PENDIENTE
                    continue

                log_fn(f"\n── Lote {completados}/{total}  ({len(batch_original)} elementos) ──")
                try:
                    resultados_batch, logs, batch_ok, raw = future.result()
                    for line in logs:
                        log_fn(line)

                    if not batch_ok:
                        # Batch fallido por reintentos agotados — activar parada y registrar motivo
                        stop_event.set()
                        hubo_fallo  = True
                        razon_fallo = (
                            f"Lote {completados}/{total} ({len(batch_original)} elementos) "
                            f"falló tras 3 intentos consecutivos.\n\n"
                            f"Respuesta cruda de Ollama:\n"
                            f"{raw[:150]}"
                            f"{'...' if len(raw) > 150 else ''}"
                        )
                        with _cache_lock:
                            for descripcion in batch_original:
                                cache_clasificaciones[descripcion] = Centinela.FALLIDO
                        log_fn("⛔ Extracción detenida. Guardando archivos parciales…")
                        if eta_callback:
                            eta_callback(None)   # limpiar ETA al detenerse por error
                    else:
                        # Batch exitoso: poblar caché con los resultados
                        with _cache_lock:
                            for descripcion, categoria in zip(batch_original, resultados_batch):
                                cache_clasificaciones[descripcion] = categoria

                except Exception as e:
                    # Error inesperado (no capturado dentro de extraer_batch)
                    log_fn(f"Error inesperado en lote {completados}: {e}")
                    stop_event.set()
                    hubo_fallo  = True
                    razon_fallo = f"Error inesperado en lote {completados}/{total}: {e}"
                    with _cache_lock:
                        for descripcion in batch_original:
                            cache_clasificaciones[descripcion] = Centinela.FALLIDO
                    if eta_callback:
                        eta_callback(None)   # limpiar ETA al detenerse por error

        # Marcar como pendientes los únicos que no llegaron a procesarse
        with _cache_lock:
            for v in unicos_pendientes:
                if v not in cache_clasificaciones:
                    cache_clasificaciones[v] = Centinela.PENDIENTE

    else:
        # Si no había nada pendiente (todo ya estaba en caché), marcar progreso como completo
        if progress_callback:
            progress_callback(1, 1)
        if eta_callback:
            eta_callback(None)   # nada que procesar → limpiar ETA

    # Resolver cada valor original usando la caché ya completa
    with _cache_lock:
        resultados = ['' if v is None else cache_clasificaciones.get(v, Centinela.PENDIENTE)
                      for v in valores_norm]
    return resultados, hubo_fallo, razon_fallo


# ─────────────────────────────────────────────
#  INTERFAZ GRÁFICA
# ─────────────────────────────────────────────

class App(tk.Tk):
    """
    Ventana principal de la aplicación.

    Secciones de la UI (de arriba a abajo):
      1. Header navy con título.
      2. Tarjeta principal con:
         - Campo API Key (con toggle mostrar/ocultar).
         - Selector de archivo Excel/CSV.
         - Combo de columna seleccionada.
         - Área de texto del prompt (editable, restaurable).
      3. Botones "Iniciar extracción" y "Detener" (fila compartida).
      4. Barra de progreso determinista.
      5. Label de tiempo restante estimado (ETA).
      6. Log de actividad en tiempo real.

    _build_ui() se divide en métodos privados por sección para reducir la longitud
    de cualquier método individual y facilitar el mantenimiento:
      _build_api_key_section(card)
      _build_file_section(card)
      _build_column_section(card)
      _build_priorizacion_section(card)
      _build_prompt_section(card)
      _build_action_buttons()
      _build_progress_and_log()

    El guardado de resultados se delega a métodos dedicados:
      _guardar_resultado_completo(df_trabajo, nombre_base, carpeta)
      _guardar_resultado_parcial(df_trabajo, nombre_base, carpeta, hubo_fallo, razon_fallo)

    Bloqueo de la UI durante el procesamiento:
      Al iniciar la extracción (_iniciar) se deshabilitan todos los controles
      de entrada (API key, checkbox mostrar, botón buscar archivo, combo de columna)
      para impedir que el usuario modifique los parámetros mientras hay un proceso
      en curso. En _finalizar() se restauran todos a su estado normal.
    """

    def __init__(self):
        super().__init__()
        self.title("Extractor de significados en textos")
        self.resizable(True, True)
        self.minsize(476, 600)
        self.configure(bg="#f0f4f8")
        self._center_window(476, 600)

        # Estado interno
        self.df      = None   # DataFrame cargado desde el archivo seleccionado
        self.archivo = None   # Ruta completa del archivo seleccionado
        self.config  = cargar_config()

        # Evento de parada compartido entre la UI y los hilos de extracción.
        # Se limpia en _iniciar() y se activa desde _detener() o cuando un batch falla.
        self._stop_event = threading.Event()

        self._build_styles()
        self._build_ui()

        # Interceptar el botón de cierre del sistema operativo para que los hilos
        # de extracción en curso reciban la señal de parada antes de destruir
        # la ventana, igual que cuando el usuario pulsa "Detener".
        self.protocol("WM_DELETE_WINDOW", self._cerrar)

    def _center_window(self, w: int, h: int) -> None:
        """Posiciona la ventana en el centro de la pantalla al abrirse."""
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x  = (sw - w) // 2
        y  = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    # ── Estilos ttk ──────────────────────────

    def _build_styles(self) -> None:
        """
        Define la paleta de colores y los estilos ttk de toda la aplicación.
        Los colores se almacenan en self._colors para reutilizarlos en _build_ui.
        """
        s = ttk.Style(self)
        s.theme_use("clam")

        # Paleta de colores
        BG     = "#f0f4f8"  # Fondo general de la ventana
        CARD   = "#ffffff"  # Fondo de la tarjeta principal
        NAVY   = "#0c2e52"  # Header y log
        ORANGE = "#e5521a"  # Acentos y botón principal
        BLUE   = "#1a6496"  # Botones secundarios y foco
        RED    = "#c0392b"  # Botón de detener
        FG     = "#1a2e44"  # Texto principal
        FG2    = "#5a7a9a"  # Texto secundario / subtítulos
        BORDER = "#d0dce8"  # Bordes de inputs
        INPUT  = "#eef2f7"  # Fondo de campos editables

        s.configure("TFrame",        background=CARD)
        s.configure("TLabel",        background=CARD,  foreground=FG,  font=("Segoe UI", 8))
        s.configure("Sub.TLabel",    background=CARD,  foreground=FG2, font=("Segoe UI", 7))
        s.configure("TCombobox",     fieldbackground=INPUT, foreground=FG,
                    background=INPUT, selectbackground=BLUE, selectforeground="#fff",
                    borderwidth=0)
        s.map("TCombobox",
              fieldbackground=[("readonly", INPUT)],
              foreground=[("readonly", FG)])
        s.configure("Start.TButton", background=ORANGE, foreground="white",
                    font=("Segoe UI", 9, "bold"), padding=(12, 5), borderwidth=0)
        s.map("Start.TButton",
              background=[("active", "#c44412"), ("disabled", "#b0b8c4")])
        # Botón Detener: rojo, se habilita sólo mientras hay una extracción en curso
        s.configure("Stop.TButton",  background=RED, foreground="white",
                    font=("Segoe UI", 9, "bold"), padding=(12, 5), borderwidth=0)
        s.map("Stop.TButton",
              background=[("active", "#922b21"), ("disabled", "#b0b8c4")])
        s.configure("File.TButton",  background=BLUE, foreground="white",
                    font=("Segoe UI", 7), padding=(5, 0), borderwidth=0)
        s.map("File.TButton",
              background=[("active", "#145080")])
        s.configure("TProgressbar",  troughcolor=BORDER, background=ORANGE,
                    borderwidth=0, thickness=5)

        self._colors = dict(BG=BG, CARD=CARD, NAVY=NAVY, ORANGE=ORANGE,
                            BLUE=BLUE, RED=RED, FG=FG, FG2=FG2, BORDER=BORDER, INPUT=INPUT)

    # ── Helpers de construcción de UI ────────

    def _section_label(self, parent, row: int, text: str) -> None:
        """Crea un label de sección (negrita navy) en la fila indicada del grid."""
        tk.Label(parent, text=text,
                 bg=self._colors["CARD"], fg=self._colors["NAVY"],
                 font=("Segoe UI", 8, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(5, 1))

    def _uniform_frame(self, parent, row: int) -> tk.Frame:
        """
        Crea un frame contenedor de altura y ancho fijos (INPUT_H x INPUT_W)
        para los campos de una sola línea.
        grid_propagate(False) evita que los widgets hijos alteren el tamaño.
        """
        f = tk.Frame(parent, bg=self._colors["CARD"], height=INPUT_H, width=INPUT_W)
        f.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 2))
        f.columnconfigure(0, weight=1)
        f.grid_propagate(False)
        return f

    # ── Secciones de la UI ───────────────────

    def _build_api_key_section(self, card: ttk.Frame) -> None:
        """
        Construye el campo de API Key con checkbox para alternar visibilidad
        y la etiqueta informativa de guardado automático.

        Se guardan referencias a self.key_entry y self.chk_show_key para poder
        deshabilitarlos durante el procesamiento y rehabilitarlos al finalizar.
        """
        c = self._colors
        self._section_label(card, 1, "🔑  API Key de Ollama")
        key_f = self._uniform_frame(card, 2)
        self.key_var = tk.StringVar(value=self.config.get("api_key", ""))
        self.key_entry = tk.Entry(key_f, textvariable=self.key_var, show="•",
                                  bg=c["INPUT"], fg=c["FG"], insertbackground=c["FG"],
                                  relief="flat", font=("Segoe UI", 8), bd=0,
                                  highlightthickness=1,
                                  highlightbackground=c["BORDER"],
                                  highlightcolor=c["BLUE"])
        self.key_entry.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        # Checkbox para alternar entre texto oculto y visible en el campo de API key.
        # Se guarda en self.chk_show_key para poder deshabilitarlo durante el proceso.
        self.show_key    = tk.BooleanVar(value=False)
        self.chk_show_key = tk.Checkbutton(key_f, text="Mostrar", variable=self.show_key,
                                           command=self._toggle_key,
                                           bg=c["CARD"], fg=c["FG2"],
                                           activebackground=c["CARD"], activeforeground=c["FG"],
                                           selectcolor=c["INPUT"], relief="flat",
                                           font=("Segoe UI", 8))
        self.chk_show_key.grid(row=0, column=1, sticky="ns")

        ttk.Label(card, text="Se guardará automáticamente para la próxima sesión.",
                  style="Sub.TLabel").grid(row=3, column=0, columnspan=2,
                                           sticky="w", pady=(1, 2))

    def _build_file_section(self, card: ttk.Frame) -> None:
        """
        Construye el selector de archivo con etiqueta de nombre/filas
        y el botón "Buscar…".

        Se guarda referencia a self.btn_file para poder deshabilitarlo durante
        el procesamiento e impedir que el usuario cambie el archivo en curso.
        """
        c = self._colors
        self._section_label(card, 4, "📂  Archivo Excel o CSV")
        file_f = self._uniform_frame(card, 5)
        self.archivo_var = tk.StringVar(value="Ningún archivo seleccionado")
        tk.Label(file_f, textvariable=self.archivo_var,
                 bg=c["INPUT"], fg=c["FG2"],
                 font=("Segoe UI", 8), anchor="w", padx=6,
                 highlightthickness=1,
                 highlightbackground=c["BORDER"]).grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        # Referencia guardada para deshabilitar/habilitar en _iniciar/_finalizar
        self.btn_file = ttk.Button(file_f, text="Buscar…", style="File.TButton",
                                   command=self._seleccionar_archivo)
        self.btn_file.grid(row=0, column=1, sticky="ns")

    def _build_column_section(self, card: ttk.Frame) -> None:
        """
        Construye el combo de selección de columna.
        Se habilita en modo "readonly" una vez que se carga un archivo válido,
        y se deshabilita completamente durante el procesamiento.
        """
        self._section_label(card, 6, "📋  Columna seleccionada")
        col_f = self._uniform_frame(card, 7)
        self.col_var = tk.StringVar()
        # El combo arranca deshabilitado; se activa al cargar un archivo válido
        self.col_combo = ttk.Combobox(col_f, textvariable=self.col_var,
                                      state="disabled", font=("Segoe UI", 8))
        self.col_combo.grid(row=0, column=0, sticky="nsew")

    def _build_priorizacion_section(self, card: ttk.Frame) -> None:
        """
        Construye el boton que abre el popup de busqueda avanzada (filtro
        opcional por columna y valor). El estado elegido se guarda en
        self.prioridad_var / self.valor_filtro_var y se resume en un label
        junto al boton para que el usuario sepa si hay un filtro activo.
        """
        c = self._colors
        self.prioridad_var = tk.StringVar(value="")
        self.valor_filtro_var = tk.StringVar(value="")
        self.filtro_status_var = tk.StringVar(value="Sin filtro")
        self.desde_var = tk.StringVar(value="")
        self.hasta_var = tk.StringVar(value="")

        adv_f = tk.Frame(card, bg=c["CARD"])
        adv_f.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(4, 2))

        self.btn_busqueda_avanzada = ttk.Button(
            adv_f, text="🔍  Búsqueda avanzada", style="File.TButton",
            command=self._abrir_busqueda_avanzada)
        self.btn_busqueda_avanzada.pack(side="left")

        tk.Label(adv_f, textvariable=self.filtro_status_var,
                 bg=c["CARD"], fg=c["FG2"], font=("Segoe UI", 7)).pack(side="left", padx=(8, 0))
        self.generar_grafico_var = tk.BooleanVar(value=False)
        self.chk_grafico = tk.Checkbutton(
            card, text="📊  Generar gráfico de resultados al finalizar",
            variable=self.generar_grafico_var,
            bg=c["CARD"], fg=c["FG"],
            activebackground=c["CARD"], activeforeground=c["FG"],
            selectcolor=c["INPUT"], relief="flat", font=("Segoe UI", 8))
        self.chk_grafico.grid(row=9, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def _abrir_busqueda_avanzada(self) -> None:
        """
        Abre un popup (Toplevel) para elegir una columna y un valor especifico
        por el cual filtrar las filas antes de extraer. Los cambios solo se
        guardan si el usuario presiona "Aplicar"; "Cancelar" descarta todo.
        """
        if self.df is None:
            messagebox.showwarning("Sin archivo", "Primero selecciona un archivo.")
            return

        c = self._colors
        popup = tk.Toplevel(self)
        popup.title("Búsqueda avanzada")
        popup.configure(bg=c["CARD"])
        popup.resizable(False, False)
        popup.transient(self)
        popup.grab_set()

        tk.Label(popup, text="⭐  Filtrar por columna", bg=c["CARD"], fg=c["NAVY"],
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, columnspan=2,
                                                      sticky="w", padx=12, pady=(12, 2))

        columnas = self.df.columns.tolist()
        col_var_local = tk.StringVar(value=self.prioridad_var.get() or "(ninguna)")
        col_combo = ttk.Combobox(popup, textvariable=col_var_local,
                                 values=["(ninguna)"] + columnas,
                                 state="readonly", font=("Segoe UI", 8), width=40)
        col_combo.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12)

        tk.Label(popup, text="🔎  Valor a extraer", bg=c["CARD"], fg=c["NAVY"],
                 font=("Segoe UI", 9, "bold")).grid(row=2, column=0, columnspan=2,
                                                      sticky="w", padx=12, pady=(10, 2))

        valor_var_local = tk.StringVar(value=self.valor_filtro_var.get() or "")
        valor_combo = ttk.Combobox(popup, textvariable=valor_var_local,
                                   state="disabled", font=("Segoe UI", 8), width=40)
        valor_combo.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12)
        tk.Label(popup, text="📐  Rango de filas (opcional)", bg=c["CARD"], fg=c["NAVY"],
                 font=("Segoe UI", 9, "bold")).grid(row=4, column=0, columnspan=2,
                                                      sticky="w", padx=12, pady=(10, 2))

        rango_f = tk.Frame(popup, bg=c["CARD"])
        rango_f.grid(row=5, column=0, columnspan=2, sticky="ew", padx=12)

        desde_var_local = tk.StringVar(value=self.desde_var.get())
        hasta_var_local = tk.StringVar(value=self.hasta_var.get())

        tk.Label(rango_f, text="Desde:", bg=c["CARD"], fg=c["FG"],
                 font=("Segoe UI", 8)).pack(side="left")
        tk.Entry(rango_f, textvariable=desde_var_local, width=6,
                 font=("Segoe UI", 8)).pack(side="left", padx=(4, 12))
        tk.Label(rango_f, text="Hasta:", bg=c["CARD"], fg=c["FG"],
                 font=("Segoe UI", 8)).pack(side="left")
        tk.Entry(rango_f, textvariable=hasta_var_local, width=6,
                 font=("Segoe UI", 8)).pack(side="left", padx=(4, 0))

        def _refrescar_valores(event=None):
            columna = col_var_local.get()
            if not columna or columna == "(ninguna)":
                valor_combo["values"] = []
                valor_var_local.set("")
                valor_combo.config(state="disabled")
                return
            valores = sorted(self.df[columna].dropna().astype(str).unique().tolist())
            valor_combo["values"] = ["(todos)"] + valores
            actual = self.valor_filtro_var.get()
            valor_var_local.set(actual if actual in valores or actual == "(todos)" else "(todos)")
            valor_combo.config(state="readonly")

        col_combo.bind("<<ComboboxSelected>>", _refrescar_valores)
        if col_var_local.get() != "(ninguna)":
            _refrescar_valores()

        def _aplicar():
            self.prioridad_var.set(col_var_local.get())
            self.valor_filtro_var.set(valor_var_local.get())
            self.desde_var.set(desde_var_local.get().strip())
            self.hasta_var.set(hasta_var_local.get().strip())
            partes = []
            if col_var_local.get() and col_var_local.get() != "(ninguna)":
                partes.append(f"{col_var_local.get()} = {valor_var_local.get()}")
            if desde_var_local.get().strip() or hasta_var_local.get().strip():
                partes.append(f"filas [{desde_var_local.get().strip() or '1'}:{hasta_var_local.get().strip() or 'fin'}]")
            self.filtro_status_var.set("🔎 Filtro activo: " + " | ".join(partes) if partes else "Sin filtro aplicado")

        def _cancelar():
            popup.destroy()

        btn_row = tk.Frame(popup, bg=c["CARD"])
        btn_row.grid(row=6, column=0, columnspan=2, sticky="ew", padx=12, pady=(12, 12))
        ttk.Button(btn_row, text="Aplicar", style="Start.TButton",
                  command=_aplicar).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(btn_row, text="Cancelar", style="File.TButton",
                  command=_cancelar).pack(side="left", expand=True, fill="x", padx=(4, 0))

        popup.columnconfigure(0, weight=1)
        popup.update_idletasks()
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - popup.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - popup.winfo_height()) // 2
        popup.geometry(f"+{x}+{y}")

    def _build_prompt_section(self, card: ttk.Frame) -> None:
        """
        Construye el área de texto del prompt con cabecera y botón
        "↺ Restaurar" que vuelve al DEFAULT_PROMPT.
        Carga el prompt guardado en sesiones previas si existe.
        """
        c = self._colors
        prompt_hdr = tk.Frame(card, bg=c["CARD"])
        prompt_hdr.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(6, 1))
        tk.Label(prompt_hdr, text="💬  Prompt (editable)",
                 bg=c["CARD"], fg=c["FG"],
                 font=("Segoe UI", 8, "bold")).pack(side="left")
        # Botón para restaurar el prompt al valor por defecto (DEFAULT_PROMPT)
        # Referencia guardada para deshabilitar/habilitar en _iniciar/_finalizar
        self.btn_restaurar = tk.Button(prompt_hdr, text="↺ Restaurar",
                                       bg=c["INPUT"], fg=c["BLUE"],
                                       font=("Segoe UI", 7), relief="flat", bd=0,
                                       cursor="hand2", padx=4,
                                       command=self._restaurar_prompt)
        self.btn_restaurar.pack(side="right")

        # Marco con borde para el área de texto del prompt
        prompt_border = tk.Frame(card, bg=c["BORDER"], bd=1)
        prompt_border.grid(row=11, column=0, columnspan=2, sticky="nsew", pady=(0, 4))
        prompt_inner = tk.Frame(prompt_border, bg=c["BORDER"])
        prompt_inner.pack(fill="both", expand=True)
        sbp = ttk.Scrollbar(prompt_inner)
        sbp.pack(side="right", fill="y")
        self.prompt_text = tk.Text(prompt_inner, height=7,
                                   bg=c["INPUT"], fg=c["FG"],
                                   insertbackground=c["FG"], relief="flat",
                                   font=("Courier New", 7), padx=6, pady=4,
                                   wrap="word", highlightthickness=0,
                                   yscrollcommand=sbp.set)
        self.prompt_text.pack(fill="both", expand=True, side="left")
        sbp.config(command=self.prompt_text.yview)
        # Cargar el prompt guardado o el por defecto si no hay sesión previa
        self.prompt_text.insert("1.0", self.config.get("ultimo_prompt", DEFAULT_PROMPT))

    def _build_action_buttons(self) -> None:
        """
        Construye la fila con los botones "Iniciar extracción" y "Detener".
        Detener arranca deshabilitado y sólo se activa durante una extracción.
        """
        c = self._colors
        btn_row = tk.Frame(self, bg=c["BG"])
        btn_row.pack(fill="x", padx=14, pady=(4, 0))

        self.btn_start = ttk.Button(btn_row, text="▶  Iniciar Extracción",
                                    style="Start.TButton", command=self._iniciar,
                                    width=26)
        self.btn_start.pack(side="left", expand=True)

        self.btn_stop = ttk.Button(btn_row, text="⏹  Detener",
                                   style="Stop.TButton", command=self._detener,
                                   width=14, state="disabled")
        self.btn_stop.pack(side="left", expand=True)

    def _build_progress_and_log(self) -> None:
        """
        Construye la barra de progreso determinista, el label de ETA y el área
        de log de actividad en tiempo real con fondo navy y scrollbar.

        El label de ETA (self.eta_var) permanece vacío en reposo y muestra el
        tiempo restante estimado mientras hay una extracción activa.
        Se actualiza desde _actualizar_eta() vía after() para ser thread-safe.
        """
        c = self._colors

        # ── Barra de progreso determinista por lotes ──
        self.progress = ttk.Progressbar(self, mode="determinate",
                                        style="TProgressbar", maximum=100, value=0)
        self.progress.pack(fill="x", padx=14, pady=(4, 0))

        # ── Label de tiempo restante estimado (ETA) ──────────────────────────
        # StringVar vacía en reposo; se rellena con el texto formateado durante
        # la extracción y se limpia al terminar en _finalizar().
        self.eta_var = tk.StringVar(value="")
        self.eta_label = tk.Label(
            self,
            textvariable=self.eta_var,
            bg=c["BG"],
            fg=c["FG2"],
            font=("Segoe UI", 7),
            anchor="center",
        )
        self.eta_label.pack(fill="x", padx=14, pady=(1, 0))
        # ─────────────────────────────────────────────────────────────────────

        # ── Log de actividad ──
        log_outer = tk.Frame(self, bg=c["BG"])
        log_outer.pack(fill="both", expand=True, padx=14, pady=(4, 10))
        tk.Label(log_outer, text="💻 Registro de actividad",
                 bg=c["BG"], fg=c["FG2"],
                 font=("Segoe UI", 7, "bold")).pack(anchor="w")
        log_border = tk.Frame(log_outer, bg=c["BORDER"], bd=1)
        log_border.pack(fill="both", expand=True, pady=(2, 0))
        self.log_box = tk.Text(log_border, height=9,
                               bg=c["NAVY"], fg="#7ab4d8",
                               insertbackground=c["FG"], relief="flat",
                               font=("Courier New", 7), state="disabled",
                               padx=6, pady=4)
        sb = ttk.Scrollbar(log_border)
        sb.pack(side="right", fill="y")
        self.log_box.pack(fill="both", expand=True, side="left")
        self.log_box.configure(yscrollcommand=sb.set)
        sb.config(command=self.log_box.yview)

    # ── Constructor principal de UI ──────────

    def _build_ui(self) -> None:
        """Construye y posiciona todos los widgets de la interfaz delegando en métodos de sección."""
        c = self._colors

        # ── Header navy ──
        header = tk.Frame(self, bg=c["NAVY"], height=44)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="Extractor de significados en textos",
                 bg=c["NAVY"], fg="white",
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=16)
        tk.Label(header, text="powered by Ollama",
                 bg=c["NAVY"], fg="#7ab4d8",
                 font=("Segoe UI", 7)).pack(side="left")

        # Franja naranja decorativa bajo el header
        tk.Frame(self, bg=c["ORANGE"], height=4).pack(fill="x")

        # ── Tarjeta principal ──
        card = ttk.Frame(self, padding=(14, 8, 14, 4))
        card.pack(fill="both", expand=True, padx=14, pady=6)
        card.columnconfigure(0, weight=1)
        card.rowconfigure(11, weight=1)  # la fila del prompt se expande al redimensionar

        # Separador naranja en la parte superior de la tarjeta
        tk.Frame(card, bg=c["ORANGE"], height=2).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        self._build_api_key_section(card)
        self._build_file_section(card)
        self._build_column_section(card)
        self._build_priorizacion_section(card)
        self._build_prompt_section(card)

        self._build_action_buttons()
        self._build_progress_and_log()

    # ── Callbacks de la UI ───────────────────

    def _restaurar_prompt(self) -> None:
        """Reemplaza el contenido del área de prompt con DEFAULT_PROMPT."""
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", DEFAULT_PROMPT)
        self._log("Prompt restaurado al original.")

    def _toggle_key(self) -> None:
        """Alterna la visibilidad del campo de API key entre texto plano y puntos."""
        self.key_entry.config(show="" if self.show_key.get() else "•")

    def _seleccionar_archivo(self) -> None:
        """
        Abre un diálogo para elegir un archivo Excel o CSV.
        Al cargarlo correctamente, actualiza el DataFrame, la etiqueta de archivo
        y el combo de columnas.
        FIX: eliminada comilla simple espuria en "*.xlsm' *.xlsb" que rompía el filtro.
        """
        ruta = filedialog.askopenfilename(
            title="Selecciona el archivo",
            filetypes=[
                ("Archivos tabulares",
                 "*.xlsx *.xls *.xlsm *.xlsb *.odf *.ods *.odt *.csv *.tsv *.txt"),
                ("Todos", "*.*"),
            ]
        )
        if not ruta:
            return  # El usuario canceló el diálogo
        try:
            self.df = cargar_dataframe(ruta)
            self.archivo = ruta
            nombre = os.path.basename(ruta)
            self.archivo_var.set(f"  {nombre}  ({len(self.df)} filas)")
            columnas = self.df.columns.tolist()
            self.col_combo["values"] = columnas
            self.col_combo.set(columnas[0])
            self.col_combo.config(state="readonly")
            self.prioridad_var.set("")
            self.valor_filtro_var.set("")
            self.filtro_status_var.set("Sin filtro")
            self._log(f"Archivo cargado: {nombre}  |  {len(self.df)} filas  |  {len(columnas)} columnas")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el archivo:\n{e}")

    def _log(self, msg: str) -> None:
        """
        Escribe un mensaje en el log de actividad de forma thread-safe.
        Usa after(0, ...) para que la actualización ocurra en el hilo principal de Tkinter,
        ya que _proceso() corre en un hilo separado.
        """
        def _write():
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(0, _write)

    def _actualizar_eta(self, segundos: float | None) -> None:
        """
        Actualiza el label de ETA de forma thread-safe usando after().

        Parámetros:
            segundos : tiempo restante estimado en segundos, o None para limpiar el label.

        Formato de salida (ejemplos):
            "⏱  Tiempo restante estimado: 2h 14m 30s"
            "⏱  Tiempo restante estimado: 4m 5s"
            "⏱  Tiempo restante estimado: 48s"
            ""   (cuando segundos es None → limpia el label)

        Se llama desde clasificar_con_cache() en el hilo de fondo, por lo que
        SIEMPRE se encola en el bucle de eventos de Tkinter mediante after().
        """
        def _write():
            if segundos is None:
                self.eta_var.set("")
                return
            secs  = int(segundos)
            horas = secs // 3600
            mins  = (secs % 3600) // 60
            segs  = secs % 60

            if horas > 0:
                texto = f"{horas}h {mins}m {segs}s"
            elif mins > 0:
                texto = f"{mins}m {segs}s"
            else:
                texto = f"{segs}s"

            self.eta_var.set(f"⏱  Tiempo restante estimado: {texto}")

        self.after(0, _write)

    def _iniciar(self) -> None:
        """
        Valida los campos del formulario y, si todo es correcto, lanza
        el proceso de extracción en un hilo separado para no bloquear la UI.
        También persiste la configuración actual antes de iniciar.

        Bloqueo de controles de entrada:
            Para evitar que el usuario modifique parámetros mientras hay un proceso
            activo, se deshabilitan todos los controles de entrada:
              - key_entry      : campo de texto de la API key.
              - chk_show_key   : checkbox "Mostrar" de la API key.
              - btn_file       : botón "Buscar…" para cambiar el archivo.
              - col_combo      : combo de selección de columna.
            Se rehabilitan en _finalizar() al terminar el proceso.
        """
        self._log("Botón 'Iniciar extracción' presionado.")

        # Validaciones de entrada
        api_key = self.key_var.get().strip()
        if not api_key:
            messagebox.showwarning("Falta la API Key", "Por favor ingresa tu API Key de Ollama.")
            self._log("Error: API Key vacía.")
            return
        if self.df is None:
            messagebox.showwarning("Sin archivo", "Por favor selecciona un archivo primero.")
            self._log("Error: Ningún archivo seleccionado.")
            return
        columna = self.col_var.get()
        if not columna:
            messagebox.showwarning("Sin columna", "Por favor selecciona la columna.")
            self._log("Error: Ninguna columna seleccionada.")
            return

        # Limpiar caché y resetear el evento de parada para esta nueva sesión
        cache_clasificaciones.clear()
        self._stop_event.clear()
        self._log("Caché de extracciones limpiado.")

        # Persistir configuración actual
        self.config["api_key"]       = api_key
        prompt = self.prompt_text.get("1.0", "end").strip()
        self.config["ultimo_prompt"] = prompt
        guardar_config(self.config)

        # ── Bloquear todos los controles de entrada durante el procesamiento ──
        # Impide que el usuario cambie la API key, el archivo, la columna o el
        # prompt mientras hay una extracción en curso, evitando inconsistencias.
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")       # habilitar Detener durante el proceso
        self.key_entry.config(state="disabled")    # bloquear campo API key
        self.chk_show_key.config(state="disabled") # bloquear checkbox "Mostrar"
        self.btn_file.config(state="disabled")     # bloquear botón "Buscar…"
        self.col_combo.config(state="disabled")    # bloquear combo de columna
        self.btn_busqueda_avanzada.config(state="disabled")
        self.chk_grafico.config(state="disabled")
        self.prompt_text.config(state="disabled")  # bloquear área de texto del prompt
        self.btn_restaurar.config(state="disabled")# bloquear botón "↺ Restaurar"
        # ─────────────────────────────────────────────────────────────────────

        self.progress.config(mode="determinate", value=0, maximum=100)
        self.eta_var.set("")                        # limpiar ETA de sesiones previas
        self._log("\n─── Iniciando extracción ───")

        threading.Thread(target=self._proceso,
                         args=(api_key, columna, prompt),
                         daemon=True).start()

    def _detener(self) -> None:
        """
        Solicita la parada cooperativa de la extracción en curso.

        Activa self._stop_event, que es comprobado por extraer_batch antes de cada
        llamada al LLM y por clasificar_con_cache antes de procesar cada batch.
        Las llamadas ya en vuelo a Ollama terminarán normalmente; la parada es efectiva
        en el siguiente punto de control (siguiente intento o siguiente batch).
        Los resultados parciales se guardarán en dos archivos separados al terminar.
        """
        self._stop_event.set()
        self._log("⏹ Detención solicitada por el usuario. Esperando que termine el batch actual…")
        # Deshabilitar inmediatamente para evitar pulsaciones múltiples mientras se frena
        self.btn_stop.config(state="disabled")

    def _cerrar(self) -> None:
        """
        Manejador del botón de cierre del sistema operativo (×).

        Garantiza que ningún hilo de extracción quede huérfano al cerrar la
        ventana: activa stop_event exactamente igual que _detener(), de modo que
        extraer_batch abandone en su próximo punto de control cooperativo.

        Los hilos son daemon=True, por lo que el intérprete los termina
        automáticamente al salir, pero activar el evento primero evita que una
        llamada en vuelo a Ollama intente escribir en widgets ya destruidos
        (lo que causaría TclError o RuntimeError en el hilo de fondo).
        """
        self._stop_event.set()   # señal de parada idéntica a "Detener"
        self.destroy()           # cerrar la ventana y terminar el mainloop

    # ── Guardado de resultados ───────────────

    def _guardar_resultado_completo(
        self,
        df_trabajo: pd.DataFrame,
        nombre_base: str,
        carpeta: str,
    ) -> None:
        """
        Persiste el DataFrame completo como '<nombre>_Extraccion.xlsx'
        y notifica al usuario con un messagebox de éxito. Si el usuario marco
        la casilla correspondiente, tambien genera un grafico de barras con
        la distribucion de categorias extraidas.
        """
        ruta_salida = os.path.join(carpeta, f"{nombre_base}_Extraccion.xlsx")
        df_trabajo.to_excel(ruta_salida, index=False)
        self._log(f"Archivo guardado: {ruta_salida}")

        ruta_grafico = None
        if self.generar_grafico_var.get():
            try:
                ruta_grafico = os.path.join(carpeta, f"{nombre_base}_grafico.png")
                generar_grafico_resultados(df_trabajo, 'respuesta_LLM', ruta_grafico)
                self._log(f"Gráfico generado: {ruta_grafico}")
            except Exception as e:
                self._log(f"⚠ No se pudo generar el gráfico: {e}")

        mensaje_grafico = f"\nGráfico guardado en:\n{ruta_grafico}\n" if ruta_grafico else ""
        self.after(0, lambda: messagebox.showinfo(
            "¡Listo!",
            f"Extracción completada.\n\n"
            f"Archivo guardado en:\n{ruta_salida}\n"
            f"{mensaje_grafico}\n"
            f"Filas procesadas: {len(df_trabajo)}"
        ))

    def _guardar_resultado_parcial(
        self,
        df_trabajo: pd.DataFrame,
        nombre_base: str,
        carpeta: str,
        hubo_fallo: bool,
        razon_fallo: str,
    ) -> None:
        """
        Separa las filas OK de las incompletas, persiste dos archivos Excel
        y muestra un messagebox detallado con motivo, conteos y rutas.

        Cuenta los centinelas ANTES de reemplazarlos por etiquetas legibles,
        comparando directamente contra los miembros del Enum para mayor robustez.
        Distingue el motivo de la parada: hubo_fallo=True → error de reintentos;
        hubo_fallo=False → parada solicitada por el usuario.
        """
        mascara_incompleta = df_trabajo['respuesta_LLM'].isin(
            [Centinela.FALLIDO, Centinela.PENDIENTE]
        )
        df_ok         = df_trabajo[~mascara_incompleta].copy()
        df_incompleto = df_trabajo[mascara_incompleta].copy()

        # Contar centinelas ANTES de sustituirlos por etiquetas legibles.
        # Hacerlo después con str.startswith() sobre las etiquetas largas también funciona
        # pero es frágil ante futuros cambios en el texto de las etiquetas.
        n_fallido   = int((df_incompleto['respuesta_LLM'] == Centinela.FALLIDO).sum())
        n_pendiente = int((df_incompleto['respuesta_LLM'] == Centinela.PENDIENTE).sum())
        n_total_inc = n_fallido + n_pendiente

        # Reemplazar centinelas internos por etiquetas legibles en el Excel de salida
        df_incompleto['respuesta_LLM'] = df_incompleto['respuesta_LLM'].replace({
            Centinela.FALLIDO:   'FALLIDO (3 intentos sin resultado válido)',
            Centinela.PENDIENTE: 'PENDIENTE (lote cancelado antes de procesar)',
        })

        ruta_ok         = os.path.join(carpeta, f"{nombre_base}_Extraccion_ok.xlsx")
        ruta_incompleto = os.path.join(carpeta, f"{nombre_base}_pendientes.xlsx")

        df_ok.to_excel(ruta_ok, index=False)
        df_incompleto.to_excel(ruta_incompleto, index=False)

        self._log(
            f"Archivos guardados: OK={len(df_ok)} | "
            f"Fallidos={n_fallido} | Pendientes={n_pendiente}"
        )

        # Construir mensaje según el motivo de la parada
        if hubo_fallo:
            titulo = "Extracción detenida por error"
            motivo = razon_fallo
        else:
            titulo = "Extracción detenida por el usuario"
            motivo = "El usuario presionó el botón Detener."

        msg = (
            f"⏹  {titulo}\n"
            f"{'─' * 52}\n\n"
            f"MOTIVO:\n{motivo}\n\n"
            f"{'─' * 52}\n\n"
            f"Se generaron DOS archivos en:\n{carpeta}\n\n"
            f"✅  Extracciones completadas correctamente  ({len(df_ok)} filas)\n"
            f"    {os.path.basename(ruta_ok)}\n\n"
            f"⚠️  Incompletas  ({n_total_inc} filas)\n"
            f"    {os.path.basename(ruta_incompleto)}\n"
            f"    • Fallidas (3 intentos, todo 'otro'):  {n_fallido} filas\n"
            f"    • Pendientes (lote cancelado):         {n_pendiente} filas\n\n"
            f"Consulta el registro de actividad para más detalles."
        )
        self.after(0, lambda: messagebox.showwarning(titulo, msg))

    # ── Proceso principal (hilo de fondo) ────

    def _proceso(self, api_key: str, columna: str, prompt: str) -> None:
        """
        Ejecuta la extracción completa en un hilo de fondo.

        Flujo en caso de ÉXITO COMPLETO:
          1. Inicializa el cliente de Ollama con la API key (local al hilo).
          2. Extrae la columna entera sobre una copia del DataFrame.
          3. Delega el guardado a _guardar_resultado_completo().
          4. Notifica al usuario con un messagebox de éxito.

        Flujo en caso de PARADA PARCIAL (por fallo de batch O por usuario):
          - Distingue el motivo: hubo_fallo=True → fallo de reintentos;
            hubo_fallo=False + stop_event activo → parada por usuario.
          - Delega el guardado y la notificación a _guardar_resultado_parcial().

        Cualquier excepción no capturada se muestra en el log y en un messagebox de error.

        FIX: client se crea localmente y se pasa como parámetro en lugar de usar
        una variable global mutable, evitando que una segunda sesión pise la primera.
        FIX: se trabaja sobre df_trabajo = self.df.copy() para no mutar self.df,
        evitando que extracciones sucesivas mezclen columnas de sesiones previas.
        """
        try:
            self._log("Configurando cliente de Ollama...")
            # FIX: client es local a este hilo; se propaga explícitamente hacia
            # clasificar_con_cache y extraer_batch, eliminando el global mutable.

            client = Client(
                host="https://ollama.com",
                headers={"Authorization": "Bearer " + api_key},
                verify=False)
            self._log("Cliente configurado correctamente.")
            self._log("Iniciando extracción...")

            # FIX: trabajar sobre una copia para no mutar self.df entre sesiones
            df_trabajo = self.df.copy()

            columna_filtro = self.prioridad_var.get()
            valor_filtro   = self.valor_filtro_var.get()
            if columna_filtro and columna_filtro != "(ninguna)":
                filas_antes = len(df_trabajo)
                df_trabajo = filtrar_por_columna(df_trabajo, columna_filtro, valor_filtro)
                self._log(f"Filtrado '{columna_filtro}' = '{valor_filtro}': {filas_antes} -> {len(df_trabajo)} filas")
            desde_str = self.desde_var.get().strip()
            hasta_str = self.hasta_var.get().strip()
            if desde_str or hasta_str:
                try:
                    desde_int = int(desde_str) if desde_str else None
                    hasta_int = int(hasta_str) if hasta_str else None
                    filas_antes_rango = len(df_trabajo)
                    df_trabajo = filtrar_por_rango(df_trabajo, desde_int, hasta_int)
                    self._log(f"Rango aplicado [{desde_str or '1'}:{hasta_str or 'fin'}]: {filas_antes_rango} -> {len(df_trabajo)} filas")
                except ValueError:
                    self._log(f"⚠ Rango invalido ('{desde_str}', '{hasta_str}'), se ignora.")

            def _batch_progress(completados: int, total: int) -> None:
                # Actualiza barra de progreso desde el hilo principal
                self.after(0, lambda: self.progress.config(maximum=total, value=completados))

            prompt_completo = prompt.rstrip() + "\n\n" + PROMPT_OBLIGATORIO.rstrip()
            resultados, hubo_fallo, razon_fallo = clasificar_con_cache(
                df_trabajo[columna].tolist(),
                prompt_completo,
                self._log,
                self._stop_event,
                client,
                OLLAMA_MODEL_ID,
                progress_callback=_batch_progress,
                eta_callback=self._actualizar_eta,   # conectar ETA a la UI
            )
            df_trabajo['respuesta_LLM'] = resultados

            nombre_base = os.path.splitext(os.path.basename(self.archivo))[0]
            carpeta     = os.path.dirname(self.archivo)

            # Determinar si hubo parada por usuario (evento activo pero sin fallo de batch)
            parada_por_usuario = self._stop_event.is_set() and not hubo_fallo

            if hubo_fallo or parada_por_usuario:
                self._guardar_resultado_parcial(
                    df_trabajo, nombre_base, carpeta, hubo_fallo, razon_fallo
                )
            else:
                self._guardar_resultado_completo(df_trabajo, nombre_base, carpeta)

        except Exception as e:
            self._log(f"\n❌  Error: {e}")
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            # Siempre restaurar la UI al estado inicial, haya éxito, parada o error
            self.after(0, self._finalizar)

    def _finalizar(self) -> None:
        """
        Restaura la UI al estado de reposo tras terminar (con éxito, parada o error).

        Acciones:
          - Detiene y resetea la barra de progreso.
          - Reactiva el botón Iniciar.
          - Deshabilita el botón Detener (no hay proceso activo).
          - Limpia el label de ETA.
          - Rehabilita todos los controles de entrada que fueron bloqueados en _iniciar():
              · key_entry    → "normal"   (campo editable).
              · chk_show_key → "normal"   (checkbox funcional).
              · btn_file     → "normal"   (botón clicable).
              · col_combo    → "readonly" (seleccionable de la lista, sin escritura libre).
                               Se usa "readonly" y no "normal" para mantener el mismo
                               comportamiento que cuando el usuario carga un archivo nuevo.
                               Si no hay archivo cargado (self.df es None) se deja
                               "disabled" para que el combo no quede habilitado vacío.
        """
        self.progress.stop()
        self.progress.config(value=0)
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.eta_var.set("")                        # limpiar ETA al finalizar

        # ── Rehabilitar controles de entrada ──────────────────────────────────
        self.key_entry.config(state="normal")       # campo API key editable de nuevo
        self.chk_show_key.config(state="normal")    # checkbox "Mostrar" funcional de nuevo
        self.btn_file.config(state="normal")        # botón "Buscar…" clicable de nuevo
        self.prompt_text.config(state="normal")     # área de texto del prompt editable de nuevo
        self.btn_restaurar.config(state="normal")   # botón "↺ Restaurar" funcional de nuevo
        # El combo vuelve a "readonly" sólo si ya hay un archivo cargado;
        # de lo contrario se mantiene "disabled" para no mostrar una lista vacía.
        if self.df is not None:
            self.col_combo.config(state="readonly")
        self.btn_busqueda_avanzada.config(state="normal")
        self.chk_grafico.config(state="normal")
        # ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
