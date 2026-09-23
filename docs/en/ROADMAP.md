# Roadmap: on-device chess recognition

## Product decision

Recognize digital chessboards and flat diagrams from imported images and a mobile
camera. Run a model locally, trained in a separate project. Keep the UI in English
and Spanish.

The camera experience should resemble QR scanning: frame the board, wait for a
stable reading, receive confirmation and review the position. Chess diagrams lack
QR error correction, so users must always be able to correct the result.

This replaces initial template-based learning. Users will not label reference
pieces before their first scan. Corrections edit the current position; they do
not automatically retrain the model.

## Current status

Implemented: Flutter scaffolding for Android, iOS and web, home screen,
localization and persisted language preference.

Pending: import, cropping, detection, model, inference, camera, position editor and
FEN export. Home actions are disabled and explicitly marked as under development.

## Scope

- Digital screenshots, diagram images and photos of diagrams on paper or screens.
- Local inference without accounts, servers or image uploads.
- Thirteen classes: empty square and six pieces of each color.
- Uncertainty review, editing of any square and orientation confirmation.
- FEN export with user-confirmed metadata.

Physical boards with three-dimensional pieces, a chess engine, general OCR,
on-device training and guaranteed universal recognition are excluded. History and
opening an external analysis site are later enhancements.

## Project responsibilities

The AI project prepares data, trains, evaluates and delivers a versioned model,
preprocessing specification, labels and reference fixtures. The app owns
image/camera input, board geometry, local integration, review and FEN.

See [MODEL_CONTRACT.md](MODEL_CONTRACT.md) for the shared handoff.

## Phase 1: independent classifier

1. Build a reproducible dataset of labeled squares and boards.
2. Combine varied digital designs and real diagrams.
3. Split training, validation and test data by source and visual style. Crops and
   augmented variants of one source image must remain in the same split.
4. Train a compact classifier and evaluate unseen styles.
5. Export a candidate mobile model, metadata and parity fixtures.

Exit criterion: reproducible per-class and exact-board results on a held-out test
set, documented limitations and an artifact ready for device testing. Training
accuracy alone is insufficient.

## Phase 2: first useful mobile workflow

1. Import a screenshot or diagram photo.
2. Allow manual crop/corner adjustment and normalize the board.
3. Extract 64 squares using the model's specified preprocessing.
4. Run the exported model locally.
5. Display the position, flag uncertain squares and allow editing any square.
6. Confirm orientation and metadata, then copy FEN.
7. Measure latency/memory on a real phone and verify operation without a network.

Exit criterion: imported image -> review -> FEN in airplane mode, without prior
user-provided reference pieces. This milestone precedes automatic scanning.

## Phase 3: detection and perspective

1. Locate the board boundary and four corners.
2. Rectify perspective and handle coordinates/margins outside the squares.
3. Reject incomplete boards and excessively blurred images.
4. Retain manual adjustment when automatic detection fails.

Exit criterion: evaluate detection and recognition together on screenshots and
diagram photos in varied conditions, including detection failures.

## Phase 4: camera and automatic capture

1. Add permissions, preview and framing guidance.
2. Sample frames without blocking the UI or queuing unbounded work.
3. Check sharpness, framing and corner stability.
4. Require agreement between successive readings and sufficient confidence.
5. Capture once, pause processing and provide visual/haptic confirmation.
6. Offer manual capture, review and retry when readings do not converge.
7. Release camera/inference resources when leaving or backgrounding the scanner.

Choose thresholds and sampling frequency from measurements. Repeating an error
across several frames does not make the result correct.

Exit criterion: stable capture with measured false captures, time to capture,
memory and energy use on representative devices.

## Phase 5: platforms and coverage

- Validate Android and iOS against the same parity fixtures.
- Integrate a local web runtime and measure its performance limits.
- Define app/model caching for offline web use after initial loading; local
  inference alone does not guarantee offline website availability.
- Broaden tests to shadows, glare, moiré, compression and annotations.
- Ship new model versions with the app and check compatibility.

Choose the first mobile platform according to available test devices. Do not
assume every Flutter package supports every target or replace local inference
with a remote API.

## Orientation, rules and FEN

The classifier returns squares in image order. The app must confirm their mapping
to chess coordinates and allow manual rotation.

A picture cannot establish the active color, castling rights, en passant target
or move counters. Request them or display editable assumptions; never present
them as recognized facts. Consistency checks may flag errors but must not silently
change pieces or invent historical information.

## Quality and risks

Measure per-class accuracy, exact-board accuracy before correction, correction
count, localization failures and end-to-end latency. Report screenshots and diagram
photos separately, as well as seen and unseen visual styles. Calibrate uncertainty
handling against validation data.

Key risks are new designs, highlighted squares, arrows, coordinates, degraded
diagrams, glare and stable but incorrect readings. Include these in the test set
instead of relying only on clean synthetic images.

Images and corrections stay local. Any future sharing of training examples must
be an explicit user action and is outside the initial workflow.
