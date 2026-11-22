import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:chess_scanner/features/scan/data/scan_image_picker.dart';

import 'package:chess_scanner/app/l10n/app_localizations.dart';
import 'package:chess_scanner/features/scan/data/boardsnap_client.dart';
import 'package:chess_scanner/features/scan/presentation/scan_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image_picker/image_picker.dart';

const placement = 'r3k3/8/8/8/8/8/8/4K2R';
void main() {
  final photoBytes = base64Decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=',
  );
  XFile photo() => XFile.fromData(photoBytes, name: 'photo.jpg');
  Future<void> openScan(WidgetTester tester, ScanPage page) async {
    await tester.pumpWidget(
      MaterialApp(
        localizationsDelegates: AppLocalizations.localizationsDelegates,
        supportedLocales: AppLocalizations.supportedLocales,
        home: page,
      ),
    );
    await tester.pumpAndSettle();
  }

  testWidgets(
    'camera opens once, previews without upload and sends photo only on Recognize',
    (tester) async {
      var shots = 0;
      var uploads = 0;
      final client = BoardSnapClient(
        baseUrl: 'https://example.test',
        clientFactory: () => MockClient((request) async {
          uploads++;
          expect(
            latin1.decode(request.bodyBytes),
            contains('name="inputType"\r\n\r\nphoto'),
          );
          return http.Response('{"piecePlacement":"$placement"}', 200);
        }),
      );
      await openScan(
        tester,
        ScanPage(
          client: client,
          initialSource: ImageSource.camera,
          imagePicker: FakeScanImagePicker(
            choose: (source) async {
              expect(source, ImageSource.camera);
              shots++;
              return shots == 1 ? photo() : null;
            },
          ),
        ),
      );
      expect(shots, 1);
      expect(uploads, 0);
      expect(find.byType(Image), findsOneWidget);
      expect(find.text('Retake photo'), findsOneWidget);
      expect(
        tester
            .widget<DropdownButtonFormField<ScanInputType>>(
              find.byType(DropdownButtonFormField<ScanInputType>),
            )
            .initialValue,
        ScanInputType.photo,
      );
      await tester.ensureVisible(find.text('Recognize'));
      await tester.tap(find.text('Recognize'));
      await tester.pumpAndSettle();
      expect(uploads, 1);
      expect(find.text(placement), findsOneWidget);
      await tester.ensureVisible(find.text('Retake photo'));
      await tester.tap(find.text('Retake photo'));
      await tester.pumpAndSettle();
      expect(shots, 2);
      expect(uploads, 1);
      expect(find.byType(Image), findsOneWidget);
      expect(find.text(placement), findsOneWidget);
    },
  );

  testWidgets(
    'cancelling the camera leaves recognition disabled and permits another attempt',
    (tester) async {
      var shots = 0;
      await openScan(
        tester,
        ScanPage(
          initialSource: ImageSource.camera,
          imagePicker: FakeScanImagePicker(
            choose: (_) async {
              shots++;
              return null;
            },
          ),
        ),
      );
      expect(shots, 1);
      expect(find.byType(Image), findsNothing);
      expect(find.byKey(const Key('scan-error')), findsNothing);
      expect(
        tester
            .widget<FilledButton>(
              find.widgetWithText(FilledButton, 'Recognize'),
            )
            .onPressed,
        isNull,
      );
      await tester.tap(find.text('Take a photo'));
      await tester.pumpAndSettle();
      expect(shots, 2);
    },
  );

  testWidgets('camera permission failure is explained and can be retried', (
    tester,
  ) async {
    var shots = 0;
    await openScan(
      tester,
      ScanPage(
        initialSource: ImageSource.camera,
        imagePicker: FakeScanImagePicker(
          choose: (_) async {
            shots++;
            if (shots == 1) {
              throw PlatformException(code: 'camera_access_denied');
            }
            return photo();
          },
        ),
      ),
    );
    expect(find.textContaining('Camera access is denied'), findsOneWidget);
    expect(find.byType(Image), findsNothing);
    expect(find.byType(LinearProgressIndicator), findsNothing);
    await tester.tap(find.text('Take a photo'));
    await tester.pumpAndSettle();
    expect(find.byKey(const Key('scan-error')), findsNothing);
    expect(find.byType(Image), findsOneWidget);
  });

  testWidgets('no available camera is handled without losing image import', (
    tester,
  ) async {
    await openScan(
      tester,
      ScanPage(
        initialSource: ImageSource.camera,
        imagePicker: FakeScanImagePicker(
          choose: (source) async {
            if (source == ImageSource.camera) {
              throw PlatformException(code: 'no_available_camera');
            }
            return photo();
          },
        ),
      ),
    );
    expect(find.byKey(const Key('scan-error')), findsOneWidget);
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose image'));
    await tester.pumpAndSettle();
    expect(find.byType(Image), findsOneWidget);
    expect(find.byKey(const Key('scan-error')), findsNothing);
  });

  testWidgets(
    'recovered camera photo retains photo mode without opening the camera again',
    (tester) async {
      var shots = 0;
      await openScan(
        tester,
        ScanPage(
          initialSource: ImageSource.camera,
          imagePicker: FakeScanImagePicker(
            recovered: RecoveredScanImage(photo(), ImageSource.camera),
            choose: (_) async {
              shots++;
              return null;
            },
          ),
        ),
      );
      expect(shots, 0);
      expect(find.byType(Image), findsOneWidget);
      expect(
        tester
            .widget<DropdownButtonFormField<ScanInputType>>(
              find.byType(DropdownButtonFormField<ScanInputType>),
            )
            .initialValue,
        ScanInputType.photo,
      );
    },
  );

  testWidgets('unsupported camera platform still offers image import', (
    tester,
  ) async {
    await openScan(
      tester,
      ScanPage(imagePicker: FakeScanImagePicker(cameraAvailable: false)),
    );
    expect(
      tester
          .widget<OutlinedButton>(
            find.widgetWithText(OutlinedButton, 'Take a photo'),
          )
          .onPressed,
      isNull,
    );
    expect(
      tester
          .widget<OutlinedButton>(
            find.widgetWithText(OutlinedButton, 'Choose image'),
          )
          .onPressed,
      isNotNull,
    );
  });

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
            imagePicker: FakeScanImagePicker(
              choose: (_) async => XFile.fromData(bytes, name: 'board.png'),
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();
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
        home: ScanPage(imagePicker: FakeScanImagePicker()),
      ),
    );
    await tester.pumpAndSettle();
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

class FakeScanImagePicker extends ScanImagePicker {
  FakeScanImagePicker({
    this.choose,
    this.recovered,
    this.cameraAvailable = true,
  });
  final Future<XFile?> Function(ImageSource)? choose;
  final RecoveredScanImage? recovered;
  final bool cameraAvailable;
  @override
  bool get supportsCamera => cameraAvailable;
  @override
  Future<XFile?> pick(ImageSource source) async =>
      choose == null ? null : await choose!(source);
  @override
  Future<RecoveredScanImage?> recover() async => recovered;
}
