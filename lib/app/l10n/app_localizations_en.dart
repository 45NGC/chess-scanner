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
  String get homeHeadline => 'Import a chessboard image';

  @override
  String get homeDescription =>
      'Scan digital chessboards and book diagrams, then convert the position into FEN for analysis.';

  @override
  String get homePrimaryAction => 'Choose image';

  @override
  String get homeSecondaryAction => 'Paste screenshot';

  @override
  String get homeComingSoon =>
      'This action will be connected in the next development step.';

  @override
  String get languageMenuTooltip => 'Change language';

  @override
  String get languageEnglish => 'English';

  @override
  String get languageSpanish => 'Spanish';
}
