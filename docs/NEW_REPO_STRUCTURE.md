# New Repo Structure

## Goal

Start the new `chess_scanner` repository from the standard Flutter project layout and add only the minimum internal structure needed to keep the app maintainable.

This version is intentionally closer to what `flutter create` generates by default.

## Recommended Base Structure

```text
chess_scanner/
  android/
  ios/
  web/
  lib/
    main.dart
    app/
      app.dart
      router/
      theme/
      l10n/
    core/
      constants/
      services/
      utils/
      error/
    features/
      home/
      import_image/
      scan/
      correction/
      history/
      settings/
    shared/
      models/
      widgets/
  test/
  docs/
    ROADMAP.es.md
    ROADMAP.en.md
    NEW_REPO_STRUCTURE.md
  l10n.yaml
  pubspec.yaml
  README.md
  analysis_options.yaml
```

## Why This Structure

- It stays aligned with the official Flutter project shape.
- It works naturally with `flutter create`.
- It avoids unnecessary monorepo complexity.
- It gives you enough room to grow inside `lib/` without turning the app into a large unstructured folder.

## What Comes From Flutter By Default

When you run `flutter create .`, the important generated parts are typically:

- `android/`
- `ios/`
- `web/`
- `lib/`
- `test/`
- `pubspec.yaml`
- `README.md`

That is the real starting point.

## What You Should Customize

The main customization should happen inside `lib/`.

Instead of keeping everything in a single `main.dart`, organize it like this:

### `lib/main.dart`

Small entrypoint only.

Responsibilities:

- call `runApp`
- bootstrap app-level configuration if needed later

### `lib/app/`

App-wide configuration.

Recommended contents:

- `app.dart`
- routing
- theme
- localization setup

### `lib/core/`

Reusable low-level utilities shared across features.

Good place for:

- constants
- small service abstractions
- image helpers
- storage helpers
- error/result types

### `lib/features/`

This is where most product code should live.

Feature folders should group UI and logic by user workflow rather than by technical type only.

Recommended initial features:

- `home/`
- `import_image/`
- `scan/`
- `correction/`
- `history/`
- `settings/`

### `lib/shared/`

Only for things genuinely shared by several features.

Examples:

- reusable widgets
- simple shared models

## Suggested Feature-Level Shape

At the beginning, keep each feature light.

Example:

```text
lib/features/scan/
  presentation/
  application/
  domain/
```

Use that only where it helps. Not every feature needs all three folders from day one.

For smaller features, this is enough:

```text
lib/features/settings/
  presentation/
```

## Recommended Initial Screens

### Home

- import image button
- recent scans
- settings access

### Import Image

- choose from gallery
- upload screenshot on web

### Scan Result

- original image preview
- detected board preview
- FEN output
- copy button
- open analysis button

### Correction

- highlighted uncertain squares
- piece picker
- group confirmation when several squares look alike

### History

- previous scans
- saved FEN positions

### Settings

- language
- reset learned templates
- import/export local data

## Internationalization

The app should be bilingual from the beginning.

Recommended setup:

```text
lib/app/l10n/
  app_en.arb
  app_es.arb
```

Recommended packages:

- `flutter_localizations`
- `intl`

Recommended behavior:

- use system locale when supported
- fallback to English otherwise
- allow manual language switching in settings

## Suggested Localization Keys

- `appTitle`
- `homeImportImage`
- `homeRecentScans`
- `scanProcessing`
- `scanResultTitle`
- `scanCopyFen`
- `scanOpenAnalysis`
- `correctionUnknownSquare`
- `correctionConfirmGroup`
- `settingsLanguage`
- `settingsResetTemplates`

## Suggested Domain Models

These do not need to exist on day one, but this is the direction:

### `ScanImageInput`

- source path or bytes
- source type
- imported at

### `DetectedBoard`

- normalized board image
- corners
- orientation
- confidence

### `SquareSample`

- row
- col
- square tone
- normalized image

### `SquareRecognition`

- row
- col
- occupied
- piece type
- piece color
- confidence

### `TemplateEntry`

- id
- visual profile id
- piece type
- piece color
- square tone
- image
- created at

### `ScanRecord`

- id
- source image
- result fen
- timestamp

## Suggested Persistence

Keep persistence simple at first.

Recommended split:

- `shared_preferences` for settings
- `isar` or `hive` for local data
- local files for stored template snapshots if needed later

## Suggested Packages

Initial candidates:

- `flutter_localizations`
- `intl`
- `go_router`
- `image_picker`
- `shared_preferences`

Later, if needed:

- `isar`
- `hive`
- `camera`
- `path_provider`
- `share_plus`

## Startup Order

1. Generate the standard Flutter app at repo root.
2. Add localization.
3. Add routing.
4. Create the base screens.
5. Implement image import.
6. Add local persistence for settings.
7. Start scan pipeline work.

## First Practical Milestone

The first useful milestone should be:

- import an image
- display it in the app
- navigate to a result screen
- prepare localization in English and Spanish

That is a much more realistic first step than overdesigning the full recognition pipeline from the beginning.
