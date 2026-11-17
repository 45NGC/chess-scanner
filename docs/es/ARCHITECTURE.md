# Arquitectura

`lib/app/` contiene inicio, tema, idioma y traducciones. `features/home/` abre el
flujo de reconocimiento.

```text
features/scan/
  data/boardsnap_client.dart       HTTP multipart, plazos, errores y validación
  domain/piece_placement.dart      lectura y generación de colocación canónica
  presentation/scan_page.dart      selección, vista previa, opciones y revisión
```

image_picker proporciona bytes mediante XFile en Android, iOS y web. Al abrir
la pantalla se recuperan selecciones perdidas por Android. Solo pulsar Reconocer
sube la imagen. Hay una petición activa como máximo y salir cierra el cliente.

BoardSnap realiza detección, perspectiva, segmentación, orientación y clasificación.
Flutter gestiona selección de modo/vista, correcciones y copia. El tablero revisado
siempre sigue el orden a8–h1. No hay modelos locales, confianza por casilla ni
etiquetado de piezas de referencia. Cámara en directo y metadatos FEN quedan pendientes.

`flutter analyze` y `flutter test` comprueban contrato multipart, errores, tiempos,
respuestas inválidas, orden de casillas, importación, edición y conservación de
correcciones al fallar. Los permisos, el selector nativo y la conexión deben
comprobarse también en dispositivos físicos.
