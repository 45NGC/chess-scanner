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
  String get homeStatusTitle => 'Base del proyecto preparada';

  @override
  String get homeStatusBody =>
      'La base oficial de Flutter ya esta lista. Los siguientes pasos son detectar el tablero, separar las casillas y corregir piezas dudosas con ayuda del usuario.';

  @override
  String get homeComingSoon =>
      'Esta accion se conectara en el siguiente paso de desarrollo.';
}
