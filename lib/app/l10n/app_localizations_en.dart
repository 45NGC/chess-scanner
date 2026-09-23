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
      'Point your camera at a digital chessboard or a printed diagram, or choose an image. Review the recognized position and export it as FEN.';

  @override
  String get homePrivacyDescription =>
      'Designed to recognize pieces on your device, without an internet connection or uploading your images. No need to teach it the pieces first.';

  @override
  String get homeDevelopmentStatus =>
      'In development: image import, camera scanning and piece recognition are not available yet.';

  @override
  String get homeCameraAction => 'Scan with camera';

  @override
  String get homeImportAction => 'Choose image';

  @override
  String get homeNextStep =>
      'First milestone: recognize a cropped board from an image on your device. Automatic camera capture will follow.';

  @override
  String get languageMenuTooltip => 'Change language';

  @override
  String get languageEnglish => 'English';

  @override
  String get languageSpanish => 'Spanish';
}
