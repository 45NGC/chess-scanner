import 'dart:convert';

import 'package:chess_scanner/app/l10n/app_localizations.dart';
import 'package:chess_scanner/features/scan/data/boardsnap_client.dart';
import 'package:chess_scanner/features/scan/presentation/scan_page.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image_picker/image_picker.dart';

const placement = 'r3k3/8/8/8/8/8/8/4K2R';
void main() {
  testWidgets(
    'non-Android screen does not attempt Android lost-data recovery',
    (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          localizationsDelegates: AppLocalizations.localizationsDelegates,
          supportedLocales: AppLocalizations.supportedLocales,
          home: ScanPage(),
        ),
      );
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('scan-error')), findsNothing);
    },
    variant: TargetPlatformVariant.only(TargetPlatform.iOS),
  );

  testWidgets(
    'imports image, reviews canonical result and preserves edits after failed retry',
    (tester) async {
      var calls = 0;
      final client = BoardSnapClient(
        baseUrl: 'https://example.test',
        clientFactory: () => MockClient((_) async {
          calls++;
          return calls == 1
              ? http.Response('{"piecePlacement":"$placement"}', 200)
              : http.Response(
                  '{"error":{"code":"BOARD_NOT_FOUND","message":"test"}}',
                  422,
                );
        }),
      );
      final bytes = base64Decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=',
      );
      await tester.pumpWidget(
        MaterialApp(
          localizationsDelegates: AppLocalizations.localizationsDelegates,
          supportedLocales: AppLocalizations.supportedLocales,
          home: ScanPage(
            client: client,
            pickImage: () async => XFile.fromData(bytes, name: 'board.png'),
          ),
        ),
      );
      await tester.tap(find.text('Choose image'));
      await tester.pumpAndSettle();
      await tester.ensureVisible(find.text('Recognize'));
      await tester.tap(find.text('Recognize'));
      await tester.pumpAndSettle();
      expect(find.text(placement), findsOneWidget);
      await tester.ensureVisible(find.byKey(const ValueKey('square-0')));
      await tester.tap(find.byKey(const ValueKey('square-0')));
      await tester.pumpAndSettle();
      await tester.tap(find.widgetWithText(OutlinedButton, 'Empty'));
      await tester.pumpAndSettle();
      const edited = '4k3/8/8/8/8/8/8/4K2R';
      expect(find.text(edited), findsOneWidget);
      await tester.ensureVisible(find.text('Recognize'));
      await tester.tap(find.text('Recognize'));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('scan-error')), findsOneWidget);
      expect(find.text(edited), findsOneWidget);
    },
  );

  testWidgets('picker cancellation does not issue a recognition request', (
    tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        localizationsDelegates: AppLocalizations.localizationsDelegates,
        supportedLocales: AppLocalizations.supportedLocales,
        home: ScanPage(pickImage: () async => null),
      ),
    );
    await tester.tap(find.text('Choose image'));
    await tester.pumpAndSettle();
    expect(find.byType(LinearProgressIndicator), findsNothing);
    expect(
      tester
          .widget<FilledButton>(find.widgetWithText(FilledButton, 'Recognize'))
          .onPressed,
      isNull,
    );
  });
}
