# Arquitectura del repositorio

## Código actual

```text
lib/
  main.dart
  app/
    app.dart
    locale/app_locale_controller.dart
    theme/app_theme.dart
    l10n/                           # Fuentes ARB y Dart generado
  features/
    home/presentation/home_page.dart
test/
  widget_test.dart
docs/
  en/
    ARCHITECTURE.md
    MODEL_CONTRACT.md
    ROADMAP.md
  es/
    ARCHITECTURE.md
    MODEL_CONTRACT.md
    ROADMAP.md
```

La pantalla inicial es la base de la interfaz del producto. Las acciones de imagen
y cámara están desactivadas hasta implementar esos flujos. La selección de idioma
sí funciona. Este repositorio todavía no contiene un motor de reconocimiento, un
archivo de modelo ni una base de datos de plantillas.

## Responsabilidades al implementar las funcionalidades

Mantener la estructura estándar de un proyecto Flutter. Añadir módulos cuando
exista código que los utilice; no crear servicios vacíos, predicciones ficticias
ni dependencias especulativas para representar puntos del plan de desarrollo.

| Módulo | Responsabilidad |
| --- | --- |
| `app/` | Inicio, tema, localización en español/inglés y preferencias de idioma |
| `features/import_image/` | Selección de imagen, vista previa y ajuste manual del recorte/esquinas |
| `features/scan/domain/` | Geometría del tablero, casillas en orden de imagen, predicciones y contrato del motor |
| `features/scan/application/` | Normalización, extracción, inferencia y composición de un resultado |
| `features/scan/data/` | Adaptadores de plataforma/ejecución, carga del modelo y preprocesamiento |
| `features/scan/presentation/` | Vista previa de cámara, ayuda de encuadre y progreso del escaneo |
| `features/correction/` | Revisión/edición de piezas, rotación de la correspondencia y confirmación de metadatos |
| `features/export/` | Validación, generación y copia de FEN |
| `features/history/` | Historial local opcional para una fase posterior |

Compartir código solo cuando varias partes lo necesiten. Conservar el significado
de las entradas y salidas del modelo entre los adaptadores móviles y web; mantener
los tipos de tensores específicos del entorno de ejecución fuera de la interfaz.

## Límites del procesamiento

Las imágenes importadas y los fotogramas seleccionados de la cámara pasan por el
mismo flujo de geometría y clasificación. El primer hito acepta un tablero
recortado manualmente. La detección automática y el muestreo continuo de cámara
llegarán después.

Tras normalizar el tablero, extraer 64 casillas en orden de filas de la imagen.
Aplicar exactamente el preprocesamiento del modelo y clasificar en 13 clases.
El adaptador debe gestionar los tamaños de lote admitidos, incluidos los modelos
que procesan una sola casilla. Mostrar puntuaciones por casilla y permitir corregir
cualquiera, incluso las predicciones con confianza alta.

El clasificador no determina el turno, los derechos históricos ni la orientación.
Asignar coordenadas de ajedrez a las posiciones de imagen solo después de confirmar
la orientación. Mantener diferenciadas las predicciones y las ediciones del usuario.

El intercambio entre proyectos se define en
[MODEL_CONTRACT.md](MODEL_CONTRACT.md).

## Ejecución y ciclo de vida

Entrenar en el proyecto Python independiente y distribuir con la app un modelo
exportado y versionado. El primer formato candidato para móvil es `.tflite`;
elegir el entorno de ejecución para Flutter después de comprobar compatibilidad,
operadores, rendimiento y plataformas con el archivo entregado. Todavía no se ha
seleccionado ni instalado ningún paquete de ejecución.

Separar el preprocesamiento y la inferencia del trabajo de interfaz cuando el
entorno lo permita. Inicializar el modelo una vez por ciclo de vida adecuado,
evitar inferencias concurrentes sin límite y liberar recursos al salir.
El escáner debe procesar el último fotograma apto en lugar de acumular una cola.
Pausar el procesamiento de cámara después de capturar y al pasar a segundo plano.

La captura automática requiere encuadre estable, calidad de imagen y acuerdo
entre lecturas. Validar los umbrales con capturas reales. Ofrecer captura manual
y reintento.

La web necesita un adaptador de ejecución local y una caché de recursos comprobada
por separado. Un complemento móvil o un modelo ejecutado localmente no garantizan
que la web pueda arrancar sin conexión. No recurrir a subir imágenes a un servicio.

## Almacenamiento y distribución

- Conservar `shared_preferences` para ajustes pequeños, como el idioma.
- Incluir el modelo inicial, sus metadatos y etiquetas como recursos de la app
  una vez validados.
- Mantener imágenes y correcciones locales, sin telemetría ni subida implícita
  de datos de entrenamiento.
- Las correcciones modifican la posición, no los pesos del modelo.
- Añadir almacenamiento de historial al implementar esa función.
- No introducir almacenes de plantillas, entrenamiento por usuario ni bases
  de datos de perfiles visuales.
- Las actualizaciones del modelo deben conservar el contrato versionado y los
  ejemplos de prueba de equivalencia.

## Interfaz y localización

La cámara y la importación de imágenes serán los puntos de entrada. El resultado
mostrará el tablero reconstruido, las dudas, la orientación y los controles de
metadatos FEN. Permitir la revisión sin exigir etiquetar piezas de referencia.

Todos los textos de interfaz pertenecen a `app_en.arb` y `app_es.arb`; regenerar
Dart con `flutter gen-l10n`. El controlador actual usa inglés por defecto y
conserva la selección manual de idioma.

## Verificación

La base actual se comprueba con `flutter analyze` y `flutter test`. Al implementar
el procesamiento, añadir comprobaciones útiles para el orden y orientación de las
casillas, la equivalencia del preprocesamiento y la ejecución, los metadatos FEN
y el ciclo de vida de los recursos de cámara. Medir la inferencia móvil real y la
captura completa por separado del entrenamiento en un ordenador.
