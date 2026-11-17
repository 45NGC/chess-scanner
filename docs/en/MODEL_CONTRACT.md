# BoardSnap integration contract

The Flutter app uses the separate Python BoardSnap HTTP service. No model is
bundled and no Python command is executed on the device. Internet/network access
is required. On-device recognition and printed diagrams are deferred.

`POST {BOARDSNAP_URL}/v1/recognize`, multipart form data:

- `image`: original image bytes, with a filename (PNG/JPEG; maximum 20 MiB).
- `inputType`: `digital` for screenshots or `photo` for photographs of screens.
- `orientation`: `white-bottom` or `black-bottom`, selected by the user. This is
  the board view, not the side to move.

The server selects the versioned model by mode. Flutter sends no model paths,
visual profiles, manual corners or FEN metadata.

HTTP 200 must contain exactly `{"piecePlacement":"..."}`. Placement is canonical
(a8 to h1), has eight ranks of eight squares, and contains only legal piece
symbols and empty runs. The client validates it before changing the current board.

Errors use `{"error":{"code":"...","message":"..."}}`. The client maps codes to
English/Spanish messages: 400 invalid request, 413 image too large, 422 invalid or
unsupported image/board not found, 500 processing/server error. Network failures,
timeouts, bad JSON and invalid placements are separate client errors. Failed
requests preserve the reviewed board. No confidence or alternative positions are
returned. Only an explicit Recognize action uploads an image.

The review board always shows White at the bottom. Every square is editable.
Copy exports only piece placement; turn, castling, en passant and counters are
still pending app features and are never inferred from the image.

The service URL is a build-time `--dart-define=BOARDSNAP_URL=...` base URL without
`/v1/recognize`. HTTPS is required outside debug builds. Web deployments need a
trusted certificate and CORS for the actual app origin. No redirect is followed
for image uploads. The request has a 60-second deadline, including reading the
response; returning from the screen closes the client. Responses are bounded to
64 KiB. The server independently validates uploads and limits concurrency.
