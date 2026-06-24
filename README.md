# chess_scanner

`chess_scanner` is a Flutter mobile and web app for importing a chessboard image, detecting the position, and exporting it as FEN for analysis.

The current repository state includes the official Flutter scaffold plus the first project-specific shell:

- bilingual app foundation in English and Spanish
- mobile and web targets
- initial home screen
- project roadmap and structure docs in `docs/`

## Flutter SDK

The Flutter SDK is installed locally at:

```bash
/home/nico/code/flutter
```

If you want to use `flutter` directly in the terminal without typing the full path each time, add this to your shell profile:

```bash
export PATH="/home/nico/code/flutter/bin:$PATH"
```

Then reload your shell:

```bash
source ~/.bashrc
```

## Common Commands

From the repository root:

```bash
/home/nico/code/flutter/bin/flutter pub get
/home/nico/code/flutter/bin/flutter analyze
/home/nico/code/flutter/bin/flutter test
/home/nico/code/flutter/bin/flutter run -d chrome
```

To see the state of your local Flutter toolchain:

```bash
/home/nico/code/flutter/bin/flutter doctor
```

## Project Structure

The app keeps the standard Flutter layout at the repository root and organizes product code inside `lib/`.

Key folders:

- `lib/app/`: app bootstrap, theme, localization
- `lib/features/`: feature-oriented UI and logic
- `docs/`: roadmap and architecture notes

See [docs/NEW_REPO_STRUCTURE.md](docs/NEW_REPO_STRUCTURE.md) for the intended evolution of the codebase.

## Next Steps

1. Build image import for gallery and web uploads.
2. Implement board detection and normalization.
3. Add square classification with confidence scoring.
4. Add user-assisted correction for uncertain squares.
5. Export validated positions as FEN.
