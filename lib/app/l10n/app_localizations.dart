import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:intl/intl.dart' as intl;

import 'app_localizations_en.dart';
import 'app_localizations_es.dart';

// ignore_for_file: type=lint

/// Callers can lookup localized strings with an instance of AppLocalizations
/// returned by `AppLocalizations.of(context)`.
///
/// Applications need to include `AppLocalizations.delegate()` in their app's
/// `localizationDelegates` list, and the locales they support in the app's
/// `supportedLocales` list. For example:
///
/// ```dart
/// import 'l10n/app_localizations.dart';
///
/// return MaterialApp(
///   localizationsDelegates: AppLocalizations.localizationsDelegates,
///   supportedLocales: AppLocalizations.supportedLocales,
///   home: MyApplicationHome(),
/// );
/// ```
///
/// ## Update pubspec.yaml
///
/// Please make sure to update your pubspec.yaml to include the following
/// packages:
///
/// ```yaml
/// dependencies:
///   # Internationalization support.
///   flutter_localizations:
///     sdk: flutter
///   intl: any # Use the pinned version from flutter_localizations
///
///   # Rest of dependencies
/// ```
///
/// ## iOS Applications
///
/// iOS applications define key application metadata, including supported
/// locales, in an Info.plist file that is built into the application bundle.
/// To configure the locales supported by your app, you’ll need to edit this
/// file.
///
/// First, open your project’s ios/Runner.xcworkspace Xcode workspace file.
/// Then, in the Project Navigator, open the Info.plist file under the Runner
/// project’s Runner folder.
///
/// Next, select the Information Property List item, select Add Item from the
/// Editor menu, then select Localizations from the pop-up menu.
///
/// Select and expand the newly-created Localizations item then, for each
/// locale your application supports, add a new item and select the locale
/// you wish to add from the pop-up menu in the Value field. This list should
/// be consistent with the languages listed in the AppLocalizations.supportedLocales
/// property.
abstract class AppLocalizations {
  AppLocalizations(String locale)
    : localeName = intl.Intl.canonicalizedLocale(locale.toString());

  final String localeName;

  static AppLocalizations? of(BuildContext context) {
    return Localizations.of<AppLocalizations>(context, AppLocalizations);
  }

  static const LocalizationsDelegate<AppLocalizations> delegate =
      _AppLocalizationsDelegate();

  /// A list of this localizations delegate along with the default localizations
  /// delegates.
  ///
  /// Returns a list of localizations delegates containing this delegate along with
  /// GlobalMaterialLocalizations.delegate, GlobalCupertinoLocalizations.delegate,
  /// and GlobalWidgetsLocalizations.delegate.
  ///
  /// Additional delegates can be added by appending to this list in
  /// MaterialApp. This list does not have to be used at all if a custom list
  /// of delegates is preferred or required.
  static const List<LocalizationsDelegate<dynamic>> localizationsDelegates =
      <LocalizationsDelegate<dynamic>>[
        delegate,
        GlobalMaterialLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
      ];

  /// A list of this localizations delegate's supported locales.
  static const List<Locale> supportedLocales = <Locale>[
    Locale('en'),
    Locale('es'),
  ];

  /// No description provided for @appTitle.
  ///
  /// In en, this message translates to:
  /// **'Chess Scanner'**
  String get appTitle;

  /// No description provided for @homeHeadline.
  ///
  /// In en, this message translates to:
  /// **'Scan a chess position'**
  String get homeHeadline;

  /// No description provided for @homeDescription.
  ///
  /// In en, this message translates to:
  /// **'Take a photo of a virtual chessboard or choose an existing image. Recognize the position with BoardSnap and review the pieces.'**
  String get homeDescription;

  /// No description provided for @homePrivacyDescription.
  ///
  /// In en, this message translates to:
  /// **'Recognition requires an internet connection and sends the selected image to your configured BoardSnap server.'**
  String get homePrivacyDescription;

  /// No description provided for @homeDevelopmentStatus.
  ///
  /// In en, this message translates to:
  /// **'Photo capture on Android/iOS, image import and piece review are available. Live scanning, complete FEN export and printed diagrams are not available yet.'**
  String get homeDevelopmentStatus;

  /// No description provided for @homeCameraAction.
  ///
  /// In en, this message translates to:
  /// **'Take a photo'**
  String get homeCameraAction;

  /// No description provided for @homeImportAction.
  ///
  /// In en, this message translates to:
  /// **'Choose image'**
  String get homeImportAction;

  /// No description provided for @homeNextStep.
  ///
  /// In en, this message translates to:
  /// **'Take a photo or choose an image, indicate which side is at the bottom, then recognize and review the position.'**
  String get homeNextStep;

  /// No description provided for @languageMenuTooltip.
  ///
  /// In en, this message translates to:
  /// **'Change language'**
  String get languageMenuTooltip;

  /// No description provided for @languageEnglish.
  ///
  /// In en, this message translates to:
  /// **'English'**
  String get languageEnglish;

  /// No description provided for @languageSpanish.
  ///
  /// In en, this message translates to:
  /// **'Spanish'**
  String get languageSpanish;

  /// No description provided for @scanTitle.
  ///
  /// In en, this message translates to:
  /// **'Recognize with BoardSnap'**
  String get scanTitle;

  /// No description provided for @scanUploadNotice.
  ///
  /// In en, this message translates to:
  /// **'Pressing Recognize sends this image to the configured BoardSnap server. Phone photographs are experimental; review every piece.'**
  String get scanUploadNotice;

  /// No description provided for @scanChooseInputType.
  ///
  /// In en, this message translates to:
  /// **'Select screenshot or phone photo'**
  String get scanChooseInputType;

  /// No description provided for @scanInputType.
  ///
  /// In en, this message translates to:
  /// **'Image type'**
  String get scanInputType;

  /// No description provided for @scanDigital.
  ///
  /// In en, this message translates to:
  /// **'Screenshot'**
  String get scanDigital;

  /// No description provided for @scanPhoto.
  ///
  /// In en, this message translates to:
  /// **'Phone photo of a screen'**
  String get scanPhoto;

  /// No description provided for @scanOrientation.
  ///
  /// In en, this message translates to:
  /// **'Board view'**
  String get scanOrientation;

  /// No description provided for @scanWhiteBottom.
  ///
  /// In en, this message translates to:
  /// **'White at the bottom'**
  String get scanWhiteBottom;

  /// No description provided for @scanBlackBottom.
  ///
  /// In en, this message translates to:
  /// **'Black at the bottom'**
  String get scanBlackBottom;

  /// No description provided for @scanOrientationHelp.
  ///
  /// In en, this message translates to:
  /// **'This describes the image view, not whose turn it is.'**
  String get scanOrientationHelp;

  /// No description provided for @scanRecognize.
  ///
  /// In en, this message translates to:
  /// **'Recognize'**
  String get scanRecognize;

  /// No description provided for @scanWorking.
  ///
  /// In en, this message translates to:
  /// **'Processing… You can go back to cancel.'**
  String get scanWorking;

  /// No description provided for @scanReview.
  ///
  /// In en, this message translates to:
  /// **'Review the position (White at the bottom). Tap a square to change its piece.'**
  String get scanReview;

  /// No description provided for @scanEmpty.
  ///
  /// In en, this message translates to:
  /// **'Empty'**
  String get scanEmpty;

  /// No description provided for @scanCopy.
  ///
  /// In en, this message translates to:
  /// **'Copy piece placement'**
  String get scanCopy;

  /// No description provided for @scanCopied.
  ///
  /// In en, this message translates to:
  /// **'Piece placement copied'**
  String get scanCopied;

  /// No description provided for @scanConfigurationError.
  ///
  /// In en, this message translates to:
  /// **'Configure BOARDSNAP_URL with the server address before starting the app.'**
  String get scanConfigurationError;

  /// No description provided for @scanBoardNotFound.
  ///
  /// In en, this message translates to:
  /// **'No supported complete board was found. Try a clearer image with the whole board visible.'**
  String get scanBoardNotFound;

  /// No description provided for @scanImageError.
  ///
  /// In en, this message translates to:
  /// **'The image could not be processed. Choose a valid PNG or JPEG with a complete virtual board.'**
  String get scanImageError;

  /// No description provided for @scanSizeError.
  ///
  /// In en, this message translates to:
  /// **'The image exceeds 20 MiB. Choose a smaller image.'**
  String get scanSizeError;

  /// No description provided for @scanTimeout.
  ///
  /// In en, this message translates to:
  /// **'The server took too long. Try again.'**
  String get scanTimeout;

  /// No description provided for @scanNetworkError.
  ///
  /// In en, this message translates to:
  /// **'Could not reach BoardSnap. Check your connection and server address, then try again.'**
  String get scanNetworkError;

  /// No description provided for @scanResponseError.
  ///
  /// In en, this message translates to:
  /// **'BoardSnap returned an unexpected response. Check that the server uses the supported contract.'**
  String get scanResponseError;

  /// No description provided for @scanPickerError.
  ///
  /// In en, this message translates to:
  /// **'Could not open the image. Check photo access and try again.'**
  String get scanPickerError;

  /// No description provided for @scanServerError.
  ///
  /// In en, this message translates to:
  /// **'BoardSnap could not process the request. Try again later.'**
  String get scanServerError;

  /// No description provided for @scanRetake.
  ///
  /// In en, this message translates to:
  /// **'Retake photo'**
  String get scanRetake;

  /// No description provided for @scanCameraPermission.
  ///
  /// In en, this message translates to:
  /// **'Camera access is denied or restricted. Allow access in your device settings, or choose an existing image.'**
  String get scanCameraPermission;

  /// No description provided for @scanCameraUnavailable.
  ///
  /// In en, this message translates to:
  /// **'Taking photos is available in the Android and iOS app when a camera is available. You can choose an existing image instead.'**
  String get scanCameraUnavailable;
}

class _AppLocalizationsDelegate
    extends LocalizationsDelegate<AppLocalizations> {
  const _AppLocalizationsDelegate();

  @override
  Future<AppLocalizations> load(Locale locale) {
    return SynchronousFuture<AppLocalizations>(lookupAppLocalizations(locale));
  }

  @override
  bool isSupported(Locale locale) =>
      <String>['en', 'es'].contains(locale.languageCode);

  @override
  bool shouldReload(_AppLocalizationsDelegate old) => false;
}

AppLocalizations lookupAppLocalizations(Locale locale) {
  // Lookup logic when only language code is specified.
  switch (locale.languageCode) {
    case 'en':
      return AppLocalizationsEn();
    case 'es':
      return AppLocalizationsEs();
  }

  throw FlutterError(
    'AppLocalizations.delegate failed to load unsupported locale "$locale". This is likely '
    'an issue with the localizations generation tool. Please file an issue '
    'on GitHub with a reproducible sample app and the gen-l10n configuration '
    'that was used.',
  );
}
