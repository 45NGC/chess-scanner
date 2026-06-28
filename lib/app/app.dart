import 'package:flutter/material.dart';

import 'locale/app_locale_controller.dart';
import '../features/home/presentation/home_page.dart';
import 'l10n/app_localizations.dart';
import 'theme/app_theme.dart';

class ChessScannerApp extends StatefulWidget {
  const ChessScannerApp({super.key});

  @override
  State<ChessScannerApp> createState() => _ChessScannerAppState();
}

class _ChessScannerAppState extends State<ChessScannerApp> {
  late final AppLocaleController _localeController;

  @override
  void initState() {
    super.initState();
    _localeController = AppLocaleController()..load();
  }

  @override
  void dispose() {
    _localeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _localeController,
      builder: (context, child) {
        return MaterialApp(
          debugShowCheckedModeBanner: false,
          locale: _localeController.locale,
          onGenerateTitle: (context) => AppLocalizations.of(context)!.appTitle,
          theme: AppTheme.light(),
          supportedLocales: AppLocalizations.supportedLocales,
          localizationsDelegates: AppLocalizations.localizationsDelegates,
          home: HomePage(localeController: _localeController),
        );
      },
    );
  }
}
