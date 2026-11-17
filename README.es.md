# Chess Scanner

[English](README.md) | Español

Cliente Flutter de **BoardSnap** para Android, iOS y web. Permite importar una
captura o fotografía de un tablero virtual, indicar modo y orientación, reconocer
la posición en el servidor Python, corregir cualquier casilla y copiar la
colocación de piezas. Incluye interfaz en inglés y español.

Necesita conexión con BoardSnap. Solo se sube la imagen al pulsar Reconocer.
El reconocimiento fotográfico sigue siendo experimental. No incluye tableros
físicos 3D, diagramas impresos, reconocimiento sin conexión, cámara en directo ni
exportación de FEN completo. No contiene pesos del modelo.

## Probar en local

En el repositorio **boardsnap**, con el extra HTTP y los modelos instalados:

```bash
.venv/bin/boardsnap-http --config data/manifests/local-trial-v1.json --model-root . --host 127.0.0.1 --port 8000 --allow-origin http://127.0.0.1:8080
```

En **chess-scanner**, desde otra terminal:

```bash
flutter pub get
flutter run -d chrome --web-hostname 127.0.0.1 --web-port 8080 --dart-define=BOARDSNAP_URL=http://127.0.0.1:8000
```

Elegir imagen → indicar captura/fotografía → blancas/negras abajo → Reconocer →
revisar. El resultado se muestra siempre con blancas abajo. Toca una casilla para
cambiar la pieza. Copiar devuelve únicamente colocación, no un FEN completo.
Si un nuevo intento falla, se conserva la posición revisada.

La URL base no debe incluir `/v1/recognize`. Si falta, se muestra un error de
configuración. En el emulador Android usa `http://10.0.2.2:8000` en debug. En un
móvil físico utiliza una dirección accesible: `localhost` representa el propio
teléfono. En iOS ya está incluida la descripción de acceso a la fototeca; utiliza
HTTPS con certificado de confianza. No se han añadido excepciones ATS generales.

HTTP se admite para pruebas **debug** (permiso Android limitado a debug). En
release se exige HTTPS. Web necesita certificado de confianza y CORS para su
origen exacto. La app no desactiva la validación de certificados. El despliegue,
la autenticación y los límites de acceso del servidor son tareas aparte.

## Comprobaciones

```bash
flutter gen-l10n
flutter analyze
flutter test
flutter build web --dart-define=BOARDSNAP_URL=https://TU-SERVIDOR
```

Comparación opcional con CLI y servidor reales, usando las imágenes de desarrollo
existentes en BoardSnap y con el servidor arrancado:

```bash
flutter test test/boardsnap_live_test.dart --dart-define=BOARDSNAP_LIVE_URL=http://127.0.0.1:8000 --dart-define=BOARDSNAP_REPO=/home/nico/code/github/boardsnap
```

Sin ambas variables se omiten esas pruebas. Comparan una captura y un fallo de
detección fotográfica con la CLI; son regresiones, no evaluación independiente.
Quedan las comprobaciones del selector nativo, permisos y conexión en dispositivos
Android/iOS y navegador real.

## Estructura

- `lib/app/`: inicio, tema, idioma y traducciones ARB/Dart generadas.
- `lib/features/home/`: entrada al flujo.
- `lib/features/scan/data/`: cliente HTTP.
- `lib/features/scan/domain/`: colocación canónica.
- `lib/features/scan/presentation/`: importación, opciones y revisión.
- `test/`: pruebas del cliente, widgets y servicio real opcional.

Documentación: [arquitectura](docs/es/ARCHITECTURE.md),
[contrato HTTP](docs/es/MODEL_CONTRACT.md), [hoja de ruta](docs/es/ROADMAP.md).
Edita los ARB y ejecuta `flutter gen-l10n`; no edites traducciones generadas.
