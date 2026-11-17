import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:chess_scanner/features/scan/data/boardsnap_client.dart';
import 'package:chess_scanner/features/scan/domain/piece_placement.dart';

const placement = 'r3k3/8/8/8/8/8/8/4K2R';
final bytes = Uint8List.fromList([137, 80, 78, 71]);
Future<String> recognize(BoardSnapClient c) => c.recognize(
  bytes: bytes,
  filename: 'board.png',
  inputType: ScanInputType.photo,
  orientation: BoardOrientation.blackBottom,
);
Matcher code(String value) =>
    isA<BoardSnapException>().having((e) => e.code, 'code', value);

void main() {
  test('placement preserves square order and does not invent FEN metadata', () {
    final squares = decodePiecePlacement(placement);
    expect(squares, hasLength(64));
    expect(squares[0], 'r');
    expect(squares[60], 'K');
    expect(squares[63], 'R');
    expect(encodePiecePlacement(squares), placement);
    expect(
      () => decodePiecePlacement('$placement w - - 0 1'),
      throwsFormatException,
    );
    expect(
      () => decodePiecePlacement('8/8/8/8/8/8/8/7'),
      throwsFormatException,
    );
  });

  test(
    'multipart sends bytes, mode and explicit orientation to correct endpoint',
    () async {
      final c = BoardSnapClient(
        baseUrl: 'https://example.test/api/',
        clientFactory: () => MockClient((r) async {
          expect(r.url.toString(), 'https://example.test/api/v1/recognize');
          expect(r.method, 'POST');
          expect(r.headers['content-type'], startsWith('multipart/form-data;'));
          final body = latin1.decode(r.bodyBytes);
          expect(body, contains('name="inputType"\r\n\r\nphoto'));
          expect(body, contains('name="orientation"\r\n\r\nblack-bottom'));
          expect(body, contains('name="image"; filename="board.png"'));
          expect(r.bodyBytes, containsAllInOrder(bytes));
          return http.Response(jsonEncode({'piecePlacement': placement}), 200);
        }),
      );
      expect(await recognize(c), placement);
    },
  );

  for (final status in [400, 413, 422, 500]) {
    test('structured error $status is preserved', () async {
      final c = BoardSnapClient(
        baseUrl: 'https://example.test',
        clientFactory: () => MockClient(
          (_) async => http.Response(
            '{"error":{"code":"BOARD_NOT_FOUND","message":"test"}}',
            status,
          ),
        ),
      );
      await expectLater(recognize(c), throwsA(code('BOARD_NOT_FOUND')));
    });
  }
  for (final body in [
    '<html>proxy error</html>',
    '{"piecePlacement":"8"}',
    '{"piecePlacement":"$placement","confidence":1}',
    '{"piecePlacement":"8/8/8/8/8/8/8/44"}',
  ]) {
    test('rejects malformed or incompatible response $body', () async {
      final c = BoardSnapClient(
        baseUrl: 'https://example.test',
        clientFactory: () => MockClient((_) async => http.Response(body, 200)),
      );
      await expectLater(recognize(c), throwsA(code('INVALID_RESPONSE')));
    });
  }
  test('missing configuration is reported before upload', () async {
    await expectLater(
      recognize(BoardSnapClient(baseUrl: '')),
      throwsA(code('CONFIGURATION')),
    );
  });
  test('network failure has a stable code', () async {
    final c = BoardSnapClient(
      baseUrl: 'https://example.test',
      clientFactory: () =>
          MockClient((_) async => throw http.ClientException('offline')),
    );
    await expectLater(recognize(c), throwsA(code('NETWORK')));
  });
  test('timeout includes reading response body', () async {
    final stream = StreamController<List<int>>();
    final c = BoardSnapClient(
      baseUrl: 'https://example.test',
      timeout: const Duration(milliseconds: 10),
      clientFactory: () => MockClient.streaming(
        (_, _) async => http.StreamedResponse(stream.stream, 200),
      ),
    );
    await expectLater(recognize(c), throwsA(code('TIMEOUT')));
    await stream.close();
  });
}
