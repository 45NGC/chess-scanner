# ROADMAP

## Vision

`chess_scanner` will be a Flutter app for mobile and web that:

- receives a screenshot or image of a digital chessboard or a book diagram
- detects the board
- splits it into 64 squares
- tries to recognize the pieces automatically
- asks the user only in uncertain cases
- learns from those corrections for future scans
- supports both Spanish and English

## V1 Scope

Included:

- digital boards from sites like lichess and chess.com
- chess book diagrams
- incremental template-based learning
- assisted manual correction
- FEN export
- opening the position in an analysis board
- bilingual UI `es/en`

Not included in this phase:

- photographed physical boards
- universal recognition without prior learning
- built-in chess engine
- advanced OCR for non-board text

## Core Idea

The system will not try to know everything from day one.

It will work like this:

1. detect the board
2. extract the squares
3. compare each square against previously learned templates
4. assign a confidence score to each result
5. ask the user only when confidence is low
6. store the answers to improve future scans

## Project Principles

- real usefulness first, sophistication later
- minimize user questions
- learn by visual style
- keep detection, recognition, and learning well separated
- store persistent knowledge instead of temporary memory only
- build internationalization from the start

## Proposed Architecture

### Frontend

Flutter with support for:

- Android
- iOS
- Web

Languages:

- Spanish
- English

Initial screens:

- Home
- Import image
- Scan result
- Manual correction
- History
- Settings

### Local app engine

Internal layers:

1. `board_detector`
2. `board_normalizer`
3. `square_extractor`
4. `square_matcher`
5. `confidence_engine`
6. `interactive_labeling`
7. `template_store`
8. `fen_builder`

## Learning Model

The system will learn from user-confirmed templates.

Each template should store:

- piece type
- piece color
- square color
- visual style
- normalized square image
- source metadata

Example styles:

- `lichess_blue`
- `lichess_brown`
- `chesscom_green`
- `book_diagram_serif_01`

## Ideal Flow

1. the user loads an image
2. the app detects the board
3. the app normalizes it
4. the app splits it into 64 squares
5. the app groups visually similar squares
6. the app tries to recognize them from previous knowledge
7. the app resolves high-confidence squares automatically
8. the app asks only about uncertain groups or squares
9. the app rebuilds the final FEN
10. the app stores the new knowledge

## Question Strategy

Do not ask square by square unless it is the last resort.

Priority:

1. try automatic recognition
2. group similar squares
3. ask by group
4. propagate the answer to the full group
5. allow single-square correction

Example:

- instead of asking 8 times for white pawns
- ask once for a group of squares that seem identical

## Phase 1: Project Foundation

Goal: have the app running and the main flow defined.

1. create a new `chess_scanner` repository
2. create a Flutter app with mobile and web support
3. define the project structure
4. create basic navigation
5. create the image import flow
6. create an empty result screen
7. configure localization for `es/en`
8. define data models for:
   - board
   - square
   - template
   - scan
   - correction

## Phase 2: Board Detection

Goal: find and normalize the board correctly.

1. detect the 8x8 region
2. fix perspective when necessary
3. normalize to a fixed size
4. detect orientation
5. support:
   - clean digital screenshots
   - scanned book diagrams
6. store debug images

Exit criterion:

- the board is extracted reliably in most known cases

## Phase 3: Square Extraction

Goal: obtain 64 consistent squares.

1. split the normalized board into 64 regions
2. crop useful margins
3. classify square tone
4. generate a normalized representation for each square
5. store debug snapshots

Exit criterion:

- every square is ready for visual matching

## Phase 4: Basic Matching

Goal: recognize already learned pieces without asking.

1. implement similarity matching
2. distinguish empty vs occupied
3. distinguish piece color
4. distinguish piece type
5. compute confidence per square
6. choose the best template by visual profile

Exit criterion:

- if the style was already learned, most squares are solved automatically

## Phase 5: Interactive Learning

Goal: make the system improve with each use.

1. store user answers
2. create new templates from confirmed squares
3. avoid useless duplicates
4. score template quality
5. allow lightweight visual-profile refinement

Exit criterion:

- after several scans of the same style, the number of questions clearly drops

## Phase 6: Smart Grouping

Goal: reduce user friction.

1. group similar squares
2. ask by group before asking by individual square
3. show visual suggestions
4. apply the answer to the full group
5. allow undo

Exit criterion:

- the average number of questions per scan becomes much lower

## Phase 7: FEN and Analysis

Goal: complete the user flow.

1. build the final FEN
2. show the reconstructed board
3. copy the FEN
4. open an analysis board
5. save scan history

## Phase 8: Style Support

Goal: support several environments.

1. profiles for lichess
2. profiles for chess.com
3. profiles for book diagrams
4. automatic selection of the most likely profile
5. creation of a new profile if none fits

## Phase 9: Persistence

Goal: do not lose learned knowledge.

1. store templates locally
2. store visual profiles
3. store history
4. export and import profiles
5. back up user knowledge

## Phase 10: Quality

Main metrics:

- final FEN accuracy
- per-square accuracy
- percentage of squares solved without help
- average number of questions per scan
- average scan time

Minimum tests:

- FEN unit tests
- matching tests
- tests with known images
- tests per visual profile
- localization tests for `es/en`

## Recommended Deliverables

### V0.1

- base Flutter app
- image import
- empty result screen
- languages `es/en`

### V0.2

- board detection
- 64-square extraction
- visual debug mode

### V0.3

- simple template matching
- FEN in easy cases

### V0.4

- user questions
- template storage

### V0.5

- grouping of similar squares
- less correction friction

### V1.0

- useful support for lichess, chess.com, and some book diagrams
- history
- FEN export
- analysis opening
- stable bilingual interface

## Risks

- large visual theme changes
- last-move highlights
- arrows and annotations drawn on top of the board
- visible coordinates contaminating borders
- low-quality or skewed book diagrams

## Current Technical Decision

For this version:

- do not start with heavy AI
- use templates + confidence + incremental learning
- ask only in uncertain cases
- learn by visual profile
- keep the entire UI available in Spanish and English

## Success Criteria

The project will be successful if:

- a user can teach a visual style in only a few scans
- the app progressively reduces the number of questions
- the final FEN becomes fast and reliable for learned styles
- the same flow works well in both Spanish and English
