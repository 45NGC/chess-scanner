# Repository architecture

## Current code

```text
lib/
  main.dart
  app/
    app.dart
    locale/app_locale_controller.dart
    theme/app_theme.dart
    l10n/                           # ARB sources and generated Dart
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

The home screen is a product shell. Its image and camera actions are disabled
until those flows exist. Language selection remains functional. There is no
recognition engine, model asset or template database in this repository.

## Responsibilities as features are implemented

Keep the standard Flutter project layout. Add modules when they have a working
consumer; do not create empty services, fake predictions or speculative package
dependencies to represent roadmap items.

| Module | Responsibility |
| --- | --- |
| `app/` | Bootstrap, theme, English/Spanish localization, locale preferences |
| `features/import_image/` | Image selection, preview and manual crop/corner adjustment |
| `features/scan/domain/` | Board geometry, image-ordered squares, predictions and engine contract |
| `features/scan/application/` | Normalize, extract, infer and assemble one scan result |
| `features/scan/data/` | Platform/runtime adapters, model loading and preprocessing |
| `features/scan/presentation/` | Camera preview, framing feedback and scan progress |
| `features/correction/` | Review/edit pieces, rotate mapping and confirm position metadata |
| `features/export/` | FEN validation, generation and copy |
| `features/history/` | Optional later local scan history |

Share code only where there is a real shared consumer. Preserve model semantics
across mobile and web adapters; keep runtime-specific tensor types out of UI code.

## Processing boundary

Both imported images and selected camera frames enter the same geometry and
classification pipeline. The first milestone accepts a manually cropped board.
Automatic board detection and continuous camera sampling come later.

After normalizing a board, extract 64 squares in image row-major order. Apply the
model's exact preprocessing and classify them into 13 classes. The model adapter
must handle supported batch sizes, including batch-one models. Present per-square
scores and allow correction of every square, even high-confidence predictions.

The classifier does not determine turn, historical rights or board orientation.
Map image positions to chess coordinates only after orientation confirmation.
Keep predicted data and user edits distinct.

The shared handoff is specified in [MODEL_CONTRACT.md](MODEL_CONTRACT.md).

## Runtime and lifecycle

Train in the separate Python project and ship an exported, versioned model with
the app. The first mobile format candidate is `.tflite`; choose a Flutter runtime
only after verifying compatibility, operators, performance and platform support
against the delivered artifact. No runtime package is selected or installed yet.

Run preprocessing/inference away from UI work where the runtime permits. Initialize
the model once per appropriate lifecycle, avoid concurrent unbounded inference,
and release resources on exit. Camera scanning should process the latest eligible
frame rather than build a backlog. Pause camera processing after capture and when
the app goes into the background.

Automatic capture requires stable framing, image quality and repeated agreement.
Thresholds need validation on real captures. Provide manual capture and retry.

Web needs a local runtime adapter and separately verified asset caching. A mobile
plugin or locally executed model does not by itself provide offline web startup.
Do not fall back to uploading images to a service.

## Storage and delivery

- Keep `shared_preferences` for small settings such as language.
- Bundle the initial model, metadata and labels as app assets once validated.
- Keep source images and corrections local; no implicit telemetry or dataset upload.
- Corrections modify the position and do not mutate model weights.
- Add history storage only when implementing history.
- Do not introduce template stores, per-user training or visual-profile databases.
- Future model updates should preserve the versioned contract and parity fixtures.

## UI and localization

Camera scanning and image import are the planned entry points. The result flow
shows the reconstructed board, uncertainty, orientation and FEN metadata controls.
Allow review without requiring users to label reference pieces.

All user-facing text belongs in `app_en.arb` and `app_es.arb`; regenerate Dart
with `flutter gen-l10n`. The current locale controller defaults to English and
persists manual selection.

## Verification

The current shell is checked with `flutter analyze` and `flutter test`. As the
pipeline is implemented, add meaningful checks for square order and orientation,
preprocessing/runtime parity, FEN metadata and camera resource lifecycle. Benchmark
actual mobile inference and end-to-end capture separately from desktop training.
