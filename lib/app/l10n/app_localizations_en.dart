// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for English (`en`).
class AppLocalizationsEn extends AppLocalizations {
  AppLocalizationsEn([String locale = 'en']) : super(locale);

  @override
  String get appTitle => 'Chess Scanner';

  @override
  String get homeHeadline => 'Scan a chess position';

  @override
  String get homeDescription =>
      'Choose a screenshot or a phone photo of a virtual chessboard. Recognize the position with BoardSnap and review the pieces.';

  @override
  String get homePrivacyDescription =>
      'Recognition requires an internet connection and sends the selected image to your configured BoardSnap server.';

  @override
  String get homeDevelopmentStatus =>
      'Image import and piece review are available. Live camera scanning, complete FEN export and printed diagrams are not available yet.';

  @override
  String get homeCameraAction => 'Scan with camera';

  @override
  String get homeImportAction => 'Choose image';

  @override
  String get homeNextStep =>
      'Choose an image, indicate its type and which side is at the bottom, then review the recognized board.';

  @override
  String get languageMenuTooltip => 'Change language';

  @override
  String get languageEnglish => 'English';

  @override
  String get languageSpanish => 'Spanish';

  @override
  String get scanTitle => 'Recognize with BoardSnap';

  @override
  String get scanUploadNotice =>
      'Pressing Recognize sends this image to the configured BoardSnap server. Phone photographs are experimental; review every piece.';

  @override
  String get scanInputType => 'Image type';

  @override
  String get scanDigital => 'Screenshot';

  @override
  String get scanPhoto => 'Phone photo of a screen';

  @override
  String get scanOrientation => 'Board view';

  @override
  String get scanWhiteBottom => 'White at the bottom';

  @override
  String get scanBlackBottom => 'Black at the bottom';

  @override
  String get scanOrientationHelp =>
      'This describes the image view, not whose turn it is.';

  @override
  String get scanRecognize => 'Recognize';

  @override
  String get scanWorking => 'Processing… You can go back to cancel.';

  @override
  String get scanReview =>
      'Review the position (White at the bottom). Tap a square to change its piece.';

  @override
  String get scanEmpty => 'Empty';

  @override
  String get scanCopy => 'Copy piece placement';

  @override
  String get scanCopied => 'Piece placement copied';

  @override
  String get scanConfigurationError =>
      'Configure BOARDSNAP_URL with the server address before starting the app.';

  @override
  String get scanBoardNotFound =>
      'No supported complete board was found. Try a clearer image with the whole board visible.';

  @override
  String get scanImageError =>
      'The image could not be processed. Choose a valid PNG or JPEG with a complete virtual board.';

  @override
  String get scanSizeError =>
      'The image exceeds 20 MiB. Choose a smaller image.';

  @override
  String get scanTimeout => 'The server took too long. Try again.';

  @override
  String get scanNetworkError =>
      'Could not reach BoardSnap. Check your connection and server address, then try again.';

  @override
  String get scanResponseError =>
      'BoardSnap returned an unexpected response. Check that the server uses the supported contract.';

  @override
  String get scanPickerError =>
      'Could not open the image. Check photo access and try again.';

  @override
  String get scanServerError =>
      'BoardSnap could not process the request. Try again later.';
}
