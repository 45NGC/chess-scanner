import 'package:flutter/material.dart';

import '../../../app/l10n/app_localizations.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    final localizations = AppLocalizations.of(context)!;
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: Text(localizations.appTitle),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                localizations.homeHeadline,
                style: theme.textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 12),
              Text(
                localizations.homeDescription,
                style: theme.textTheme.bodyLarge?.copyWith(height: 1.5),
              ),
              const SizedBox(height: 24),
              FilledButton.icon(
                onPressed: () => _showComingSoon(context, localizations),
                icon: const Icon(Icons.image_search_rounded),
                label: Text(localizations.homePrimaryAction),
              ),
              const SizedBox(height: 12),
              OutlinedButton.icon(
                onPressed: () => _showComingSoon(context, localizations),
                icon: const Icon(Icons.content_paste_search_rounded),
                label: Text(localizations.homeSecondaryAction),
              ),
              const SizedBox(height: 32),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        localizations.homeStatusTitle,
                        style: theme.textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        localizations.homeStatusBody,
                        style: theme.textTheme.bodyMedium?.copyWith(height: 1.5),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showComingSoon(
    BuildContext context,
    AppLocalizations localizations,
  ) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(localizations.homeComingSoon)),
    );
  }
}
