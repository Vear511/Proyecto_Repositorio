# Extractor de significado en textos

Herramienta con interfaz gráfica que extrae resúmenes o clasificaciones de una gran cantidad de filas en una columna específica de un archivo tabular, usando un LLM a través de la API de Ollama.

## Origen del proyecto

El código base de este programa fue desarrollado durante una práctica profesional, previo al período de evaluación de COM4602. Durante el período de evaluación se agregaron las siguientes funciones nuevas:

- **Filtro por columna y valor** (búsqueda avanzada): permite extraer solo las filas donde otra columna coincide con un valor específico (ej. procesar solo las filas donde "Sucursal" = "Santiago").
- **Filtro por rango de filas**: permite restringir la extracción a un rango específico de filas (ej. filas 1 a 50), funcionando de forma independiente o combinado con el filtro por columna.
- **Gráfico de resultados**: casilla opcional que, al marcarla, genera un gráfico de barras (PNG) con la distribución de categorías extraídas al finalizar el proceso.

## Requisitos

Para utilizar el programa se necesita conexión a internet obligatoriamente, ya que las extracciones se realizan a través de la API de Ollama en línea. El ejecutable (.exe) está diseñado para correr en Windows. No requiere instalar Python, librerías ni ningún programa adicional: basta con tener el archivo .exe para usarlo.

Para correr el código fuente directamente (no el .exe) se necesita Python 3.10+ y las librerías: `pandas`, `ollama`, `matplotlib`, `tkinter` (incluido en la mayoría de instalaciones de Python).

## Aviso de seguridad de Windows

Si Windows muestra una advertencia de seguridad al abrir el ejecutable, haz clic en "Más información" y luego en "Ejecutar de todas formas". Esto ocurre porque el ejecutable no tiene firma digital, pero el programa es completamente seguro.

## API Key

Es la llave que se le da al programa para utilizar el LLM; sin ella el programa no funcionará y mostrará un error indicando que falta la API key o que es errónea. Se consigue en el sitio web de Ollama (https://ollama.com), donde hay que iniciar sesión o registrarse para obtener una API key gratuita en la sección de ajustes.

El programa guarda automáticamente la API key entre sesiones, en un archivo `.clasificador_config.json` dentro de la carpeta personal del usuario (ej. `C:\Users\TuNombre\.clasificador_config.json`). Para cambiarla, basta con escribir la nueva en el campo e iniciar una extracción.

El campo de la API key muestra el texto oculto con puntos por defecto. Se puede marcar el checkbox "Mostrar" para alternar entre texto oculto y visible.

## Archivo a utilizar

Debe ser una tabla con datos en una columna de la que se quiera extraer algún significado. Extensiones admitidas: `.csv`, `.tsv`, `.txt`, `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, `.odf`, `.ods`, `.odt`.

## Prompt y cómo usarlo

El programa viene con un prompt predeterminado orientado a clasificar activos fijos en categorías específicas (departamento, vehículo, terreno, etc.). Este prompt puede modificarse libremente para adaptarse a cualquier otra tarea de procesamiento de texto: resúmenes, extracción de datos, etiquetado, entre otros.

Por ejemplo, si se quieren clasificaciones específicas de una columna, en el prompt se deben entregar las categorías que se quieren utilizar como regla para que el LLM clasifique los datos dentro de alguna de ellas.

### Reglas obligatorias (hardcodeadas)

El programa aplica siempre estas reglas además del prompt del usuario, y no se pueden modificar sin romper el funcionamiento del código:

```
Reglas obligatorias
-. Si es ambigua, vacía o no encaja, responde 'otro'.
-. Debes responder EXACTAMENTE la misma cantidad de líneas que descripciones recibiste.
-. Respeta las tildes y caracteres especiales exactamente como están escritas.

Prohibido en la respuesta: Explicaciones, puntuación extra, comillas, mayúsculas.

Formato obligatorio:
1: [Respuesta 1]
2: [Respuesta 2]
3: [Respuesta 3]
```

Si se busca extraer un dato con palabras específicas, la categoría "otro" debe usarse para los datos que no se puedan extraer con esas categorías.

Cualquier fila vacía o con valores inválidos (NaN, vacíos, puntos solos) no pasa por el LLM y queda vacía en la columna de respuestas.

El prompt se guarda automáticamente entre sesiones. El botón "↺ Restaurar" vuelve al prompt original en cualquier momento.

## Búsqueda avanzada (filtro por columna y rango)

El botón "🔍 Búsqueda avanzada" abre un popup con dos filtros opcionales, que pueden usarse por separado o en conjunto:

- **Filtrar por columna**: elige una columna del archivo y un valor específico; solo se extraerán las filas donde esa columna coincida con ese valor. Dejar en "(ninguna)" o "(todos)" para no filtrar.
- **Rango de filas**: permite indicar un "Desde" y "Hasta" (1-indexado, inclusive) para restringir la extracción a un tramo específico del archivo. Funciona de forma independiente al filtro por columna.

El texto junto al botón resume el filtro actualmente activo. Los valores se guardan hasta que se cambien o se cargue un archivo nuevo (que resetea el filtro).

## Gráfico de resultados

La casilla "📊 Generar gráfico de resultados al finalizar" es opcional. Si se marca antes de iniciar la extracción, al terminar con éxito se genera un gráfico de barras (`<nombre>_grafico.png`) con la cantidad de filas por cada categoría extraída, guardado en la misma carpeta que el archivo original. Si la generación del gráfico falla por algún motivo, se registra en el log pero no interrumpe el guardado del archivo de resultados.

## Botón de iniciar extracción y detener

Al presionar "Iniciar Extracción" comienza el proceso. Puede detenerse en cualquier momento con el botón "Detener", aunque puede demorar en frenarse según cuánto tarde en terminar la llamada a la API que ya estaba en curso.

## Barra de progreso

Es determinista y avanza por lotes completos, no de forma continua. También se muestra un contador de tiempo restante estimado, calculado en base al promedio de duración de los lotes ya procesados.

## Log

Muestra en tiempo real qué está sucediendo: archivo cargado, filtros aplicados, datos únicos a extraer, resultados de cada lote, reintentos, errores y la finalización del proceso.

## Cómo funciona el código de extracción

Los datos se normalizan (se eliminan espacios sobrantes, valores vacíos, NaN y caracteres sin sentido) y se arma una lista de valores únicos, para no enviar datos repetidos al LLM más de una vez. Los valores se envían al LLM tal como están en el archivo original; solo la respuesta del modelo se lleva a minúsculas al parsearla.

Los datos únicos se dividen en lotes de 300 elementos (más de eso el LLM empieza a fallar por saturación) y se procesan en paralelo con 3 hilos. La API gratuita de Ollama solo atiende 2 solicitudes simultáneas; la tercera queda en fila hasta que una de las dos anteriores libere espacio, y así sucesivamente hasta agotar los lotes.

### Sistema de reintentos

Si el LLM devuelve 'otro' para todos los datos de un lote (indicando que no pudo comprenderlos), el lote se reintenta hasta 3 veces antes de marcarse como fallido. Esto se refleja en el log.

Una vez procesados, los resultados se mapean de vuelta a cada fila original según su valor único, y se agregan en una nueva columna `respuesta_LLM`.

## Archivos generados

El archivo de salida se guarda en la misma carpeta que el original:

- **Extracción completa**: `<nombre>_Extraccion.xlsx`
- **Si se detiene o hay error**: `<nombre>_Extraccion_ok.xlsx` (filas completadas) y `<nombre>_pendientes.xlsx` (filas FALLIDO o PENDIENTE)
- **Si se marcó la casilla de gráfico**: `<nombre>_grafico.png`

### Caso de detener

Si el usuario detiene el proceso, se guarda lo ya completado en `_extraccion_ok.xlsx` y lo pendiente en `_pendientes.xlsx` (filas marcadas PENDIENTE: canceladas antes de enviarse al LLM).

### Casos de error

Si ocurre un error durante la extracción (de la API, de columna, etc.), se guardan igualmente los dos archivos: `_extraccion_ok.xlsx` con lo completado, y `_pendientes.xlsx` con las filas FALLIDO (el lote que causó el error, tras agotar los 3 reintentos) y PENDIENTE (lotes que no alcanzaron a procesarse).

Otros errores posibles: falta la API key, o no se seleccionó un archivo/columna.

## Privacidad

El ejecutable corre en el computador del usuario, pero el texto de la columna seleccionada **sí se envía a la API de Ollama en la nube** para ser procesado. Se recomienda revisar las políticas de privacidad de Ollama si se manejan datos sensibles o confidenciales.

## Importante

- Si el archivo a procesar está abierto en Excel al mismo tiempo, el programa no podrá cargarlo. Ciérralo antes de seleccionarlo. Tampoco funciona si el archivo está protegido con contraseña.
- El archivo original no se borra, pero archivos con el mismo nombre que terminen en `_extraccion.xlsx`, `_extraccion_ok.xlsx`, `_pendientes.xlsx` o `_grafico.png` corren riesgo de ser sobrescritos.

```
Original.xlsx                  → Completamente a salvo
Original_extraccion.xlsx       → Riesgo de ser sobrescrito
Original_extraccion_ok.xlsx    → Riesgo de ser sobrescrito
Original_pendientes.xlsx       → Riesgo de ser sobrescrito
Original_grafico.png           → Riesgo de ser sobrescrito
```

- Si se cierra el programa mientras extrae datos (botón X u otro método), se comporta igual que al presionar "Detener": los lotes pendientes se cancelan y se guarda lo ya extraído. Mientras este guardado ocurre en segundo plano, el programa puede tardar en volver a abrirse; una vez terminado, se puede reabrir y revisar los archivos generados con normalidad.

## Ejecución

```bash
python3 Extractor_de_significado_en_textos.py
```
