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
  String get homeHeadline => 'Escanea una posición de ajedrez';

  @override
  String get homeDescription =>
      'Apunta con la cámara a un tablero digital o a un diagrama impreso, o elige una imagen. Revisa la posición reconocida y expórtala como FEN.';

  @override
  String get homePrivacyDescription =>
      'Diseñada para reconocer las piezas en tu dispositivo, sin conexión y sin subir tus imágenes. No tendrás que enseñarle las piezas antes de usarla.';

  @override
  String get homeDevelopmentStatus =>
      'En desarrollo: la importación de imágenes, el escaneo con cámara y el reconocimiento de piezas todavía no están disponibles.';

  @override
  String get homeCameraAction => 'Escanear con cámara';

  @override
  String get homeImportAction => 'Elegir imagen';

  @override
  String get homeNextStep =>
      'Primer objetivo: reconocer un tablero recortado de una imagen en tu dispositivo. Después llegará la captura automática con cámara.';

  @override
  String get languageMenuTooltip => 'Cambiar idioma';

  @override
  String get languageEnglish => 'Inglés';

  @override
  String get languageSpanish => 'Español';
}
