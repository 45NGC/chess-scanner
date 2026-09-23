# Chess Scanner

[English](README.md) | Español

Aplicación Flutter para reconocer posiciones de ajedrez en tableros digitales y
diagramas impresos, con **inferencia en el dispositivo** y exportación FEN como
objetivo del producto.

En móvil, la experiencia prevista es similar a escanear un código QR: apuntar la
cámara al tablero, esperar una lectura nítida y estable, y revisar la posición.
Las capturas y las imágenes de diagramas importadas utilizarán el mismo flujo de
reconocimiento. Los tableros físicos con piezas tridimensionales quedan fuera del
alcance.

## Estado actual

Este repositorio contiene:

- La estructura base de Flutter para Android, iOS y web.
- Una pantalla inicial en español e inglés que describe el nuevo flujo.
- Selección de idioma guardada localmente.
- Una prueba básica de la interfaz y documentación de arquitectura e integración
  del modelo.

**La importación de imágenes, la captura con cámara, la detección del tablero,
la inferencia, la corrección y la exportación FEN todavía no están implementadas.**
Las acciones de la pantalla inicial están desactivadas y se indica que están en
desarrollo. Todavía no se incluye ningún modelo entrenado ni entorno de ejecución
para inferencia.

## Enfoque de reconocimiento

El plan anterior de pedir al usuario que etiquetara piezas de referencia y guardar
plantillas visuales se ha sustituido por un modelo entrenado en otro proyecto.

- El entrenamiento y la preparación de datos se realizan fuera de esta app.
- La app incluirá un modelo versionado y ejecutará el reconocimiento localmente.
- El usuario no tendrá que enseñarle cada pieza antes de su primer escaneo.
- Las correcciones manuales modificarán la posición actual; no reentrenarán
  silenciosamente el modelo ni subirán imágenes.
- El reconocimiento debe funcionar sin servidor ni cuenta. Abrir una web externa
  de análisis será una acción opcional e independiente que requiere conexión.

El flujo previsto es:

```text
Imagen importada / fotograma seleccionado de la cámara
  -> localizar el tablero (primero mediante recorte manual)
  -> corregir la perspectiva
  -> extraer 64 casillas en orden de imagen
  -> clasificar cada casilla localmente entre 13 clases
  -> revisar las casillas dudosas y confirmar la orientación
  -> confirmar los metadatos de la posición y exportar FEN
```

El escaneo con cámara también comprobará el encuadre, la nitidez, la estabilidad
del tablero y el acuerdo entre lecturas sucesivas antes de capturar.
La confianza es una puntuación del modelo, no una garantía de acierto.

## Proyectos de la app y del modelo

| Este repositorio | Proyecto de IA independiente |
| --- | --- |
| Interfaz Flutter, importación y cámara | Preparación y procedencia de los datos |
| Encuadre, normalización y extracción de casillas | Entrenamiento y evaluación del modelo |
| Integración de la ejecución local y revisión | Modelo exportado, metadatos y ejemplos de referencia |
| Orientación, FEN y ajustes locales | Mediciones de rendimiento y comandos de entrenamiento reproducibles |

El primer modelo propuesto para móvil es un clasificador compacto en formato
`.tflite`. El entorno de ejecución, las dimensiones de los tensores y la
cuantización deben validarse con un modelo real antes de añadir dependencias.
La web sigue siendo una plataforma objetivo, pero necesita su propia integración
de ejecución local y validación de la caché de recursos; el arranque del sitio
sin conexión todavía no está implementado ni garantizado.

El contrato compartido está disponible en
[español](docs/es/MODEL_CONTRACT.md) e
[inglés](docs/en/MODEL_CONTRACT.md).

## Hitos de desarrollo

1. Crear el prototipo del clasificador independiente y evaluar estilos visuales
   no vistos durante el entrenamiento.
2. Importar/recortar una imagen estática, ejecutar el modelo exportado en un móvil
   real, revisar el tablero y exportar FEN.
3. Añadir localización automática del tablero y corrección de perspectiva.
4. Añadir vista previa de cámara, reconocimiento de fotogramas seleccionados
   y captura automática estable.
5. Validar la integración en iOS/web, el funcionamiento sin conexión y una mayor
   variedad de diagramas.

El entrenamiento y la interfaz de importación pueden avanzar de forma independiente
siguiendo el contrato documentado. Hay que medir la precisión del modelo con
ejemplos reales antes de afirmar que un tipo de imagen está soportado.

## Desarrollo

Con Flutter instalado y el comando `flutter` disponible en `PATH`, ejecutar estos
comandos desde la raíz del repositorio:

```bash
flutter pub get
flutter gen-l10n
flutter analyze
flutter test
flutter run -d chrome
```

Usar `flutter run` con un móvil conectado para probar en el dispositivo.
La interfaz y la prueba básica actuales no permiten determinar el rendimiento
de la inferencia.

## Estructura y planificación

- `lib/app/`: inicio, tema, preferencias de idioma y localización generada.
- `lib/features/home/`: pantalla inicial actual.
- `test/`: pruebas de Flutter.

La documentación está organizada por idioma, con documentos equivalentes
en cada carpeta:

| Documento | Español | Inglés |
| --- | --- | --- |
| Arquitectura | [Arquitectura](docs/es/ARCHITECTURE.md) | [Arquitectura en inglés](docs/en/ARCHITECTURE.md) |
| Contrato del modelo | [Contrato del modelo](docs/es/MODEL_CONTRACT.md) | [Contrato en inglés](docs/en/MODEL_CONTRACT.md) |
| Plan de desarrollo | [Plan de desarrollo](docs/es/ROADMAP.md) | [Plan en inglés](docs/en/ROADMAP.md) |

Editar `lib/app/l10n/app_en.arb` y `app_es.arb`, y ejecutar `flutter gen-l10n`
para regenerar los archivos Dart de localización.
