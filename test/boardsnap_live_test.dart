// Opt-in development regression: reuses inspected images, never the final set.
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:chess_scanner/features/scan/data/boardsnap_client.dart';

void main() {
  const url = String.fromEnvironment('BOARDSNAP_LIVE_URL');
  const repo = String.fromEnvironment('BOARDSNAP_REPO');
  for (final entry in {
    ScanInputType.digital:
        '.cache/real-use-tests/computer-screenshots/real-use-1.png',
    ScanInputType.photo: '.cache/real-use-tests/IMG_5526.jpg',
  }.entries) {
    test(
      'real HTTP ${entry.key.name} response agrees with pinned-model CLI',
      () async {
        final model = entry.key == ScanInputType.digital
            ? '.cache/neural-tolerance-v1-conservative/best.pt'
            : '.cache/neural-mixed-sqrt-v1/last.pt';
        final cli = await Process.run('$repo/.venv/bin/boardsnap', [
          entry.value,
          '--model',
          model,
          '--input-type',
          entry.key.name,
          '--orientation',
          'white-bottom',
        ], workingDirectory: repo);
        final expected =
            jsonDecode(cli.stdout as String) as Map<String, dynamic>;
        final client = BoardSnapClient(baseUrl: url);
        addTearDown(client.close);
        final future = client.recognize(
          bytes: await File('$repo/${entry.value}').readAsBytes(),
          filename: entry.value.split('/').last,
          inputType: entry.key,
          orientation: BoardOrientation.whiteBottom,
        );
        if (cli.exitCode == 0) {
          expect(await future, expected['piecePlacement']);
        } else {
          await expectLater(
            future,
            throwsA(
              isA<BoardSnapException>().having(
                (e) => e.code,
                'code',
                expected['error']['code'],
              ),
            ),
          );
        }
      },
      skip: url.isEmpty || repo.isEmpty,
      timeout: const Timeout(Duration(minutes: 2)),
    );
  }
}
