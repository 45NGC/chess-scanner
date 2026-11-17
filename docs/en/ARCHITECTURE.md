# Architecture

`lib/app/` owns bootstrap, theme, locale settings and English/Spanish localization.
`features/home/` opens the recognition flow.

```text
features/scan/
  data/boardsnap_client.dart       multipart HTTP, deadlines, errors, validation
  domain/piece_placement.dart      canonical placement decoding/encoding
  presentation/scan_page.dart      picker, preview, mode/view controls, review
```

Image selection uses image_picker/XFile bytes on Android, iOS and web. Android
lost picker data is recovered when opening the scan screen. The client uploads
only after the user presses Recognize, accepts a single request at a time, and
closes requests when leaving the screen. Language and HTTP code remain separate.

BoardSnap owns detection, perspective correction, segmentation, orientation and
classification. The app owns mode/view selection, user corrections and export.
No models, per-square confidence, local recognition or reference-piece labeling
are part of this integration. The review board uses canonical a8-to-h1 order.
Full FEN metadata and live camera capture remain future modules.

Run `flutter analyze` and `flutter test`. Tests cover multipart fields/bytes,
structured errors, malformed responses, timeout, square order, import, editing,
failed retries preserving corrections and picker cancellation. Native permissions,
real picker behavior and connectivity must also be checked on physical devices.
