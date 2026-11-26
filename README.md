# Chess Scanner

English | [Español](README.es.md)

Flutter client for **BoardSnap**, targeting Android, iOS and web. Take a photo on
Android/iOS or import a screenshot
or phone photograph of a virtual chessboard, select its mode and orientation,
recognize it through the Python server, review/edit every square and copy the
piece-placement field. English and Spanish UI are included.

Recognition requires a reachable BoardSnap server and uploads the selected image
only when you press Recognize. Phone recognition remains experimental. Physical
3D boards, printed diagrams, offline recognition, live camera scanning and full
FEN metadata/export are outside this implementation. No model is bundled.

## Try locally

In the **boardsnap** repository (with its HTTP extra and pinned models installed):

```bash
.venv/bin/boardsnap-http --config data/manifests/local-trial-v1.json --model-root . --host 127.0.0.1 --port 8000 --allow-origin http://127.0.0.1:8080
```

In **chess-scanner**, in another terminal:

```bash
flutter pub get
flutter run -d chrome --web-hostname 127.0.0.1 --web-port 8080 --dart-define=BOARDSNAP_URL=http://127.0.0.1:8000
```

Choose image → select screenshot or phone photo → choose White/Black at the bottom
→ Recognize → review. The resulting board is always displayed with White at the
bottom. Tap any square to change its piece. Copy exports only piece placement,
not a full FEN. A failed recognition preserves the previous reviewed position.

The base URL must omit `/v1/recognize`. A missing URL produces a configuration
error. For an Android emulator on this machine, use `http://10.0.2.2:8000` in debug.
For physical devices use a reachable server address; `localhost` means the phone
itself. iOS requires the configured photo-library usage description (included);
use trusted HTTPS for device testing. No broad ATS exception is installed.

HTTP is enabled for local **debug** testing (Android debug manifest only). Release
clients require HTTPS. Deployed web clients also need a trusted certificate and
CORS for their exact origin. The app does not disable certificate validation.
Backend deployment and authentication/rate limits remain separate work.

Both home actions open their picker directly and then use the same recognition
screen. For an imported image, explicitly select **Screenshot** or **Phone photo
of a screen** before recognition is enabled; the app does not infer the type from
the extension. A new import clears that selection. Camera photos select photo
mode automatically. Both modes send the chosen White/Black orientation through
the same HTTP client and display the canonical returned position unchanged.

## Take a photo (Android/iOS)

Tap **Take a photo** on the home screen to open the native camera. Take and accept
the photo, then review its preview. **Retake photo** opens the camera again;
cancelling preserves the current image and reviewed board. New camera photos use
**Phone photo of a screen** mode automatically. Choose the board orientation and
press **Recognize** to upload. Taking or previewing a photo never uploads it.

A device without a camera or a denied/restricted permission shows a clear error;
you can retry or choose an existing image. If access is blocked, enable it in the
device's settings. iOS includes `NSCameraUsageDescription` and the existing photo
library description; no microphone permission is needed for still photos. Android
uses image_picker's external camera intent and needs no extra CAMERA/storage
permission in the app manifest. Source metadata is stored locally only while the
picker is active, so Android can restore the image and photo/gallery mode after
process recreation when you reopen the scan screen. Photos stay in the picker
cache; this feature does not save them to a permanent gallery or scan history.

Camera capture here is native Android/iOS only. Web/desktop keep image import;
there is no browser live-camera flow. Simulator/emulator cameras and permissions
may differ from a real phone. Native capture/permissions still require device
verification. Restart the app after permission/configuration changes; hot reload
alone does not rebuild iOS configuration.

## Verification

```bash
flutter gen-l10n
flutter analyze
flutter test
flutter build web --dart-define=BOARDSNAP_URL=https://YOUR-SERVER
```

Optional real CLI/HTTP parity check, with the development server running and the
existing development fixtures in the BoardSnap repository:

```bash
flutter test test/boardsnap_live_test.dart --dart-define=BOARDSNAP_LIVE_URL=http://127.0.0.1:8000 --dart-define=BOARDSNAP_REPO=/home/nico/code/github/boardsnap
```

Live tests are skipped unless both defines are provided. They compare a screenshot
and a photographic detection failure with the CLI; they are regression checks,
not an independent model evaluation. Native picker behavior, permissions and
end-to-end networking must still be tested on Android/iOS and a real browser.

## Structure

- `lib/app/`: bootstrap, theme, locale settings, ARB sources/generated translations.
- `lib/features/home/`: entry point.
- `lib/features/scan/data/`: HTTP adapter.
- `lib/features/scan/domain/`: canonical piece placement.
- `lib/features/scan/presentation/`: import, preview, recognition and review.
- `test/`: client, widget and opt-in real-service checks.

Documentation: [architecture](docs/en/ARCHITECTURE.md),
[HTTP contract](docs/en/MODEL_CONTRACT.md), [roadmap](docs/en/ROADMAP.md).
Edit ARB files and run `flutter gen-l10n`; do not edit generated translations.
