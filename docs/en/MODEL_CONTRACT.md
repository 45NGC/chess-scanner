# App / AI model contract

Status: proposed v1 handoff for the first classifier prototype. No model has been
trained or integrated in this repository. This document fixes label semantics and
image ordering; concrete tensor details must accompany the first exported model.

## Scope and boundary

The model classifies individual squares of a flat digital board or printed diagram.
The app owns board localization, perspective correction and square extraction.
The AI project should also provide a reference CLI that takes an already cropped,
rectified 8×8 board, extracts its squares and returns predictions in image order.
This CLI is a development tool, not a server required by the app.

Initial reference images contain the board only, with no outer coordinates or
margins. Define any additional per-square crop in model metadata and fixtures.
Do not silently rotate, mirror or infer chess coordinates from the input.

## Class order

This exact zero-based order is shared by training, exported outputs and the app:

| Index | Label | FEN symbol |
| --- | --- | --- |
| 0 | empty | none (encoded as an empty run in FEN) |
| 1 | white_pawn | P |
| 2 | white_knight | N |
| 3 | white_bishop | B |
| 4 | white_rook | R |
| 5 | white_queen | Q |
| 6 | white_king | K |
| 7 | black_pawn | p |
| 8 | black_knight | n |
| 9 | black_bishop | b |
| 10 | black_rook | r |
| 11 | black_queen | q |
| 12 | black_king | k |

Piece color means chess side, independent of artwork or square background.
An uncertain result is a review state derived from scores, not a fourteenth class.
Low confidence must never be converted to an empty square.

## Square order and outputs

- Rows and columns are zero-based in image coordinates.
- Index = row × 8 + column; index 0 is top-left and 63 is bottom-right.
- This is not necessarily a8-to-h1 order: orientation belongs to the app.
- The logical result is 64 vectors of 13 class probabilities, plus model version.
- The reference CLI returns a JSON object with `model_version` and `squares`.
  Each of exactly 64 ordered entries has `row`, `column` and `probabilities`.
- Probabilities must be finite, within [0, 1], and sum to one within a documented
  numerical tolerance. The top score is not a calibrated guarantee of correctness.
- Runtime adapters dequantize outputs and apply softmax only if the artifact
  declares logits. They must not apply softmax twice.
- The physical tensor may accept one square or a batch. Document the supported
  shapes; the app may process all 64 squares in batches or successive calls.

The reference implementation must specify board splitting for dimensions not
divisible by eight, image orientation handling and resizing conventions. Supply
fixtures so Dart and Python can reproduce the same preprocessing.

## Versioned delivery bundle

Proposed contents, to be created by the AI project:

```text
export/<model-version>/
  model.tflite
  labels.json
  metadata.json
  evaluation.json
  MODEL_CARD.md
  fixtures/
```

`labels.json` is an array of the 13 strings in the order above. The metadata must
contain the contract version, model version and artifact checksum, plus:

- Input/output tensor names, shapes, layouts and dtypes; supported batch sizes.
- Square dimensions, RGB/grayscale channel order, alpha compositing, resize
  interpolation, crop policy and input normalization formula.
- Whether normalization is embedded in the model or performed by the caller.
- Quantization scales, zero points and axes when applicable.
- Whether outputs are logits or probabilities and the conversion to the logical
  probability vectors defined above.
- Model/runtime requirements, including any nonstandard operators.
- A validation-derived uncertainty policy, with its limitations.

Tensor size, architecture and quantization are intentionally undecided until
experiments establish a suitable baseline. A float model can be the first export.
Any quantized candidate needs its own accuracy and parity evaluation.

## Acceptance evidence

Provide:

1. Reproducible dataset preparation, training, evaluation and export commands,
   dependency versions, random seeds and dataset/source identifiers.
2. Data provenance, licenses and grouping rules preventing source-image/style
   leakage between training, validation and test sets.
3. Per-class metrics/confusion matrix and exact-board accuracy on held-out boards,
   separated by screenshots versus diagram photos and seen versus unseen styles.
4. Measurements before and after export/quantization, artifact size, and timing
   with hardware/runtime clearly identified. Desktop latency is not mobile latency.
5. Licensed reference images with expected preprocessing tensors or reproducible
   hashes, expected probability vectors and justified comparison tolerances.
6. A model card describing covered styles, limitations, failures and data gaps.

Keep augmented variants and all square crops of a board in the same split.
When the dataset cannot support an unseen-style test or real-diagram evaluation,
report the gap rather than presenting synthetic performance as general accuracy.

## App-owned interpretation

The app decides orientation, highlights uncertainty and supports manual editing.
It asks for or explicitly displays assumed turn, castling rights, en passant
target and move counters before exporting a complete FEN. These are not outputs
of the image classifier.

Corrections change the current position only. There is no on-device retraining,
implicit example upload or external inference service in this contract.
