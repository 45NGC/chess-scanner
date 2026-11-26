# Contrato de integración con BoardSnap

Flutter utiliza el servicio HTTP separado de BoardSnap en Python. No incluye
pesos ni ejecuta Python en el dispositivo. Se necesita conexión. El reconocimiento
local y los diagramas impresos quedan pendientes.

`POST {BOARDSNAP_URL}/v1/recognize`, formulario multipart:

- `image`: bytes originales y nombre del PNG/JPEG, máximo 20 MiB.
- `inputType`: `digital` (captura) o `photo` (fotografía de una pantalla).
- `orientation`: `white-bottom` o `black-bottom`. Describe la vista, no el turno.

El servidor selecciona el modelo por modo. Flutter no envía perfiles, rutas de
pesos, esquinas manuales ni metadatos FEN.

El éxito HTTP 200 contiene exclusivamente `{"piecePlacement":"..."}`, en orden
canónico de a8 a h1. La app valida ocho filas de ocho casillas y símbolos válidos
antes de actualizar el tablero. Los errores tienen la forma
`{"error":{"code":"...","message":"..."}}` y se muestran traducidos.

Se distinguen errores de petición (400), tamaño (413), imagen o tablero (422),
servidor (500), conexión, tiempo de espera y respuesta inválida. Los fallos
conservan la posición revisada. Solo pulsar Reconocer envía la imagen; no se
solicitan puntuaciones de confianza ni posiciones alternativas.

La revisión muestra siempre blancas abajo y permite cambiar cualquier casilla.
Copiar exporta solo la colocación; turno, enroques, captura al paso y contadores
son futuras funciones de la app y no se deducen de la imagen.

La URL base se fija con `--dart-define=BOARDSNAP_URL=...`, sin `/v1/recognize`.
HTTPS es obligatorio fuera de debug. Web necesita certificado de confianza y
CORS para el origen de la app. No se siguen redirecciones al subir imágenes.
El límite de espera es 60 segundos e incluye leer la respuesta, limitada a 64 KiB.
Salir de la pantalla cierra el cliente; el servidor podría terminar un trabajo
que ya hubiera recibido. El servidor vuelve a validar imágenes y concurrencia.

## Flujo compartido de imágenes

Cámara y galería utilizan la misma pantalla. Las fotos de cámara seleccionan
`photo`; las imágenes importadas requieren elegir explícitamente `digital` o
`photo` antes de enviarlas. Cada nueva importación borra la selección de tipo.
La extensión no decide el modo. Ambas entradas envían la orientación elegida,
conservan la colocación canónica recibida y solo suben al pulsar Reconocer.
Inicio abre el selector correspondiente una sola vez; una imagen recuperada de
Android se previsualiza sin abrir otro selector. Cancelar no envía peticiones.
