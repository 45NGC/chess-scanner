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
      'Elige una captura o una fotografía de un tablero virtual. Reconoce la posición con BoardSnap y revisa las piezas.';

  @override
  String get homePrivacyDescription =>
      'El reconocimiento necesita conexión y envía la imagen seleccionada al servidor BoardSnap configurado.';

  @override
  String get homeDevelopmentStatus =>
      'Ya puedes importar imágenes y revisar las piezas. La cámara en directo, la exportación de FEN completo y los diagramas impresos siguen pendientes.';

  @override
  String get homeCameraAction => 'Escanear con cámara';

  @override
  String get homeImportAction => 'Elegir imagen';

  @override
  String get homeNextStep =>
      'Elige una imagen, indica su tipo y qué color está abajo, y revisa el tablero reconocido.';

  @override
  String get languageMenuTooltip => 'Cambiar idioma';

  @override
  String get languageEnglish => 'Inglés';

  @override
  String get languageSpanish => 'Español';

  @override
  String get scanTitle => 'Reconocer con BoardSnap';

  @override
  String get scanUploadNotice =>
      'Al pulsar Reconocer se enviará esta imagen al servidor BoardSnap configurado. El modo fotografía es experimental; revisa todas las piezas.';

  @override
  String get scanInputType => 'Tipo de imagen';

  @override
  String get scanDigital => 'Captura de pantalla';

  @override
  String get scanPhoto => 'Fotografía de una pantalla';

  @override
  String get scanOrientation => 'Vista del tablero';

  @override
  String get scanWhiteBottom => 'Blancas abajo';

  @override
  String get scanBlackBottom => 'Negras abajo';

  @override
  String get scanOrientationHelp =>
      'Describe la vista de la imagen, no quién tiene el turno.';

  @override
  String get scanRecognize => 'Reconocer';

  @override
  String get scanWorking => 'Procesando… Puedes volver atrás para cancelar.';

  @override
  String get scanReview =>
      'Revisa la posición (blancas abajo). Toca una casilla para cambiar su pieza.';

  @override
  String get scanEmpty => 'Vacía';

  @override
  String get scanCopy => 'Copiar colocación de piezas';

  @override
  String get scanCopied => 'Colocación de piezas copiada';

  @override
  String get scanConfigurationError =>
      'Configura BOARDSNAP_URL con la dirección del servidor antes de iniciar la app.';

  @override
  String get scanBoardNotFound =>
      'No se encontró un tablero completo compatible. Prueba con una imagen más clara que muestre todo el tablero.';

  @override
  String get scanImageError =>
      'No se pudo procesar la imagen. Elige un PNG o JPEG válido con un tablero virtual completo.';

  @override
  String get scanSizeError =>
      'La imagen supera los 20 MiB. Elige una imagen más pequeña.';

  @override
  String get scanTimeout =>
      'El servidor ha tardado demasiado. Vuelve a intentarlo.';

  @override
  String get scanNetworkError =>
      'No se pudo conectar con BoardSnap. Comprueba la conexión y la dirección del servidor y vuelve a intentarlo.';

  @override
  String get scanResponseError =>
      'BoardSnap devolvió una respuesta inesperada. Comprueba que el servidor utiliza el contrato compatible.';

  @override
  String get scanPickerError =>
      'No se pudo abrir la imagen. Comprueba el acceso a las fotos y vuelve a intentarlo.';

  @override
  String get scanServerError =>
      'BoardSnap no pudo procesar la solicitud. Inténtalo más tarde.';
}
