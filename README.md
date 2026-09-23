# Chess Scanner

English | [Español](README.es.md)

Flutter app for recognizing chess positions in digital boards and printed diagrams,
with **on-device inference** and FEN export as the product goal.

On mobile, the intended experience is similar to scanning a QR code: point the
camera at a board, wait for a clear and stable reading, then review the position.
Imported screenshots and diagram images will use the same recognition pipeline.
Physical boards with three-dimensional pieces are outside the scope.

## Current status

This repository currently contains:

- Flutter scaffolding for Android, iOS and web.
- An English/Spanish home screen describing the new workflow.
- Language selection persisted locally.
- A widget smoke test and the architecture/model handoff documentation.

**Image import, camera capture, board detection, model inference, correction and
FEN export are not implemented yet.** The home actions are disabled and labeled
as under development. No trained model or inference runtime is bundled yet.

## Recognition approach

The previous plan of asking users to label reference pieces and storing visual
templates has been replaced by a model trained in a separate project.

- Training and dataset preparation happen outside this app.
- The app will ship a versioned model and run recognition locally.
- Users will not need to teach the app each piece before their first scan.
- Manual corrections will adjust the current position; they will not silently
  retrain the model or upload images.
- Recognition must work without a server or an account. Opening an external
  analysis website is a separate, optional online action.

The intended pipeline is:

```text
Image import / selected camera frame
  -> locate board (manual crop first)
  -> correct perspective
  -> extract 64 squares in image order
  -> classify each square locally into one of 13 classes
  -> review uncertain squares and confirm orientation
  -> confirm position metadata and export FEN
```

Camera scanning will additionally check framing, sharpness, board stability and
agreement across successive readings before capture. Confidence is a model score,
not a guarantee of correctness.

## App and model projects

| This repository | Separate AI project |
| --- | --- |
| Flutter UI, import and camera | Dataset preparation and provenance |
| Board framing, normalization and square extraction | Model training and evaluation |
| Local runtime integration and review | Exported model, metadata and reference fixtures |
| Orientation, FEN and local settings | Benchmarks and reproducible training commands |

The proposed first mobile artifact is a compact `.tflite` classifier. The exact
runtime, tensor dimensions and quantization must be validated with a real model
before adding dependencies. Web remains a target, but needs its own local-runtime
integration and asset caching validation; offline browser startup is not yet
implemented or guaranteed.

The shared contract is available in
[English](docs/en/MODEL_CONTRACT.md) and
[Spanish](docs/es/MODEL_CONTRACT.md).

## Development milestones

1. Build the separate classifier prototype and evaluate unseen visual styles.
2. Import/crop a static image, run the exported model on a real mobile device,
   review the board and export FEN.
3. Add automatic board location and perspective correction.
4. Add camera preview, sampled-frame recognition and stable automatic capture.
5. Validate iOS/web integration, offline behavior and broader diagram coverage.

Training and image-import UI work can progress independently using the documented
contract. Model accuracy must be measured on real examples before claiming support.

## Development

With Flutter installed and the `flutter` command available on `PATH`, run these
commands from the repository root:

```bash
flutter pub get
flutter gen-l10n
flutter analyze
flutter test
flutter run -d chrome
```

Use `flutter run` with a connected mobile device for device testing. Runtime
performance cannot be established by the current shell or widget test.

## Structure and plans

- `lib/app/`: bootstrap, theme, locale preferences and generated localization.
- `lib/features/home/`: current home screen.
- `test/`: Flutter tests.

Documentation is organized by language, with equivalent documents in each folder:

| Document | English | Español |
| --- | --- | --- |
| Architecture | [Architecture](docs/en/ARCHITECTURE.md) | [Arquitectura](docs/es/ARCHITECTURE.md) |
| Model contract | [Model contract](docs/en/MODEL_CONTRACT.md) | [Contrato del modelo](docs/es/MODEL_CONTRACT.md) |
| Roadmap | [Roadmap](docs/en/ROADMAP.md) | [Plan de desarrollo](docs/es/ROADMAP.md) |

Edit `lib/app/l10n/app_en.arb` and `app_es.arb`, then run `flutter gen-l10n` to
regenerate the Dart localization files.
