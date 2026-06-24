import 'package:flutter/material.dart';

import '../features/home/presentation/home_page.dart';
import 'l10n/app_localizations.dart';
import 'theme/app_theme.dart';

class ChessScannerApp extends StatelessWidget {
  const ChessScannerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      locale: const Locale('en'),
      onGenerateTitle: (context) => AppLocalizations.of(context)!.appTitle,
      theme: AppTheme.light(),
      supportedLocales: AppLocalizations.supportedLocales,
      localizationsDelegates: AppLocalizations.localizationsDelegates,
      home: const HomePage(),
    );
  }
}
