import 'dart:async';
import 'dart:convert';

import 'package:chess_scanner/app/l10n/app_localizations.dart';
import 'package:chess_scanner/features/scan/data/boardsnap_client.dart';
import 'package:chess_scanner/features/scan/data/scan_image_picker.dart';
import 'package:chess_scanner/features/scan/presentation/scan_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image_picker/image_picker.dart';

const placement = 'r3k3/8/8/8/8/8/8/4K2R';

class FlowImagePicker extends ScanImagePicker {
    int calls = 0;
    ImageSource? requestedSource;

    @override
    bool get supportsCamera => true;

    @override
    Future<RecoveredScanImage?> recover() async => null;

    @override
    Future<XFile?> pick(ImageSource source) async {
        calls++;
        requestedSource = source;
        return XFile.fromData(
            base64Decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='),
            name: 'image-$calls.png',
        );
    }
}

Future<void> openScan(WidgetTester tester, FlowImagePicker picker, BoardSnapClient client,
    {ImageSource source = ImageSource.gallery}) async {
    await tester.pumpWidget(MaterialApp(
        localizationsDelegates: AppLocalizations.localizationsDelegates,
        supportedLocales: AppLocalizations.supportedLocales,
        home: ScanPage(
            imagePicker: picker,
            client: client,
            initialSource: source,
            openPickerOnStart: true,
        ),
    ));
    await tester.pumpAndSettle();
}

Future<void> selectType(WidgetTester tester, ScanInputType type) async {
    final dropdown = find.byType(DropdownButtonFormField<ScanInputType>);
    await tester.ensureVisible(dropdown);
    await tester.tap(dropdown);
    await tester.pumpAndSettle();
    await tester.tap(find.text(type == ScanInputType.digital ? 'Screenshot' : 'Phone photo of a screen').last);
    await tester.pumpAndSettle();
}

Future<void> selectOrientation(WidgetTester tester, BoardOrientation orientation) async {
    final dropdown = find.byType(DropdownButtonFormField<BoardOrientation>);
    await tester.ensureVisible(dropdown);
    await tester.tap(dropdown);
    await tester.pumpAndSettle();
    await tester.tap(find.text(orientation == BoardOrientation.whiteBottom
        ? 'White at the bottom' : 'Black at the bottom').last);
    await tester.pumpAndSettle();
}

void main() {
    for (final source in ImageSource.values) {
        final types = source == ImageSource.camera ? [ScanInputType.photo] : ScanInputType.values;
        for (final type in types) {
            for (final orientation in BoardOrientation.values) {
                testWidgets('${source.name}, ${type.name}, ${orientation.wireValue} share one recognition flow', (tester) async {
                    var uploads = 0;
                    final picker = FlowImagePicker();
                    final client = BoardSnapClient(
                        baseUrl: 'https://example.test',
                        clientFactory: () => MockClient((request) async {
                            uploads++;
                            final body = latin1.decode(request.bodyBytes);
                            expect(request.url.path, '/v1/recognize');
                            expect(body, contains('name="inputType"\r\n\r\n${type.name}'));
                            expect(body, contains('name="orientation"\r\n\r\n${orientation.wireValue}'));
                            expect(body, contains('name="image"; filename='));
                            return http.Response('{"piecePlacement":"$placement"}', 200);
                        }
                        ),
                    );
                    await openScan(tester, picker, client, source: source);
                    expect(picker.calls, 1);
                    expect(picker.requestedSource, source);
                    expect(find.byType(Image), findsOneWidget);
                    expect(uploads, 0);
                    if (source == ImageSource.gallery) {
                        expect(tester.widget<FilledButton>(find.widgetWithText(FilledButton, 'Recognize')).onPressed, isNull);
                        await selectType(tester, type);
                    }
                    await selectOrientation(tester, orientation);
                    expect(uploads, 0);
                    await tester.ensureVisible(find.text('Recognize'));
                    await tester.tap(find.text('Recognize'));
                    await tester.pumpAndSettle();
                    expect(uploads, 1);
                    // The server already returns canonical order, including Black-bottom views.
                    expect(find.text(placement), findsOneWidget);
                }
                );
            }
        }
    }

    testWidgets('each new gallery image requires its own explicit mode selection', (tester) async {
        var uploads = 0;
        final picker = FlowImagePicker();
        final client = BoardSnapClient(
            baseUrl: 'https://example.test',
            clientFactory: () => MockClient((_) async {
                uploads++;
                return http.Response('{"piecePlacement":"$placement"}', 200);
            }
            ),
        );
        await openScan(tester, picker, client);
        await selectType(tester, ScanInputType.photo);
        await tester.ensureVisible(find.text('Recognize'));
        await tester.tap(find.text('Recognize'));
        await tester.pumpAndSettle();
        expect(find.text(placement), findsOneWidget);
        await tester.ensureVisible(find.text('Choose image'));
        await tester.tap(find.text('Choose image'));
        await tester.pumpAndSettle();
        expect(picker.calls, 2);
        expect(find.text(placement), findsNothing);
        expect(find.text('Select screenshot or phone photo'), findsOneWidget);
        expect(tester.widget<FilledButton>(find.widgetWithText(FilledButton, 'Recognize')).onPressed, isNull);
        expect(uploads, 1);
    }
    );

    testWidgets('both image sources and scan controls are disabled during recognition', (tester) async {
        final response = Completer<http.Response>();
        var uploads = 0;
        final picker = FlowImagePicker();
        final client = BoardSnapClient(
            baseUrl: 'https://example.test',
            clientFactory: () => MockClient((_) {
                uploads++;
                return response.future;
            }
            ),
        );
        await openScan(tester, picker, client);
        await selectType(tester, ScanInputType.digital);
        await tester.ensureVisible(find.text('Recognize'));
        await tester.tap(find.text('Recognize'));
        await tester.pump();
        expect(tester.widget<FilledButton>(find.widgetWithText(FilledButton, 'Recognize')).onPressed, isNull);
        for (final label in ['Choose image', 'Take a photo']) {
            expect(tester.widget<OutlinedButton>(find.widgetWithText(OutlinedButton, label)).onPressed, isNull);
        }
        expect(tester.widget<DropdownButtonFormField<ScanInputType>>(
            find.byType(DropdownButtonFormField<ScanInputType>)).onChanged, isNull);
        expect(tester.widget<DropdownButtonFormField<BoardOrientation>>(
            find.byType(DropdownButtonFormField<BoardOrientation>)).onChanged, isNull);
        expect(uploads, 1);
        response.complete(http.Response('{"piecePlacement":"$placement"}', 200));
        await tester.pumpAndSettle();
        expect(find.text(placement), findsOneWidget);
    }
    );
}
