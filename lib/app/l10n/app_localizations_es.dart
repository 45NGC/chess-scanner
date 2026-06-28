// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for Spanish Castilian (`es`).
class AppLocalizationsEs extends AppLocalizations {
  AppLocalizationsEs([String locale = 'es']) : super(locale);

  @override
  String get appTitle => 'Chess Scanner';

  @override
  String get homeHeadline => 'Importa una imagen del tablero';

  @override
  String get homeDescription =>
      'Escanea tableros digitales y diagramas de libros, y convierte la posicion en FEN para analizarla.';

  @override
  String get homePrimaryAction => 'Elegir imagen';

  @override
  String get homeSecondaryAction => 'Pegar captura';

  @override
  String get homeComingSoon =>
      'Esta accion se conectara en el siguiente paso de desarrollo.';

  @override
  String get languageMenuTooltip => 'Cambiar idioma';

  @override
  String get languageEnglish => 'Ingles';

  @override
  String get languageSpanish => 'Espanol';
}
