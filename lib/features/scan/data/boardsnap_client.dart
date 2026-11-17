import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../domain/piece_placement.dart';

enum ScanInputType { digital, photo }

enum BoardOrientation {
  whiteBottom('white-bottom'),
  blackBottom('black-bottom');

  const BoardOrientation(this.wireValue);
  final String wireValue;
}

class BoardSnapException implements Exception {
  const BoardSnapException(this.code);
  final String code;
}

/// Uploads original bytes; geometry, model selection and recognition stay in Python.
class BoardSnapClient {
  BoardSnapClient({
    required this.baseUrl,
    http.Client Function()? clientFactory,
    this.timeout = const Duration(seconds: 60),
  }) : _clientFactory = clientFactory ?? http.Client.new;

  final String baseUrl;
  final Duration timeout;
  final http.Client Function() _clientFactory;
  http.Client? _active;
  static const maxImageBytes = 20 * 1024 * 1024;

  Future<String> recognize({
    required Uint8List bytes,
    required String filename,
    required ScanInputType inputType,
    required BoardOrientation orientation,
  }) async {
    final base = Uri.tryParse(baseUrl);
    if (base == null ||
        base.host.isEmpty ||
        base.userInfo.isNotEmpty ||
        base.hasQuery ||
        base.hasFragment ||
        !['http', 'https'].contains(base.scheme) ||
        (!kDebugMode && base.scheme != 'https')) {
      throw const BoardSnapException('CONFIGURATION');
    }
    if (bytes.isEmpty) throw const BoardSnapException('INVALID_IMAGE');
    if (bytes.length > maxImageBytes) {
      throw const BoardSnapException('PAYLOAD_TOO_LARGE');
    }
    if (_active != null) throw const BoardSnapException('BUSY');
    final client = _clientFactory();
    _active = client;
    try {
      return await _send(
        client,
        base,
        bytes,
        filename,
        inputType,
        orientation,
      ).timeout(timeout);
    } on TimeoutException {
      throw const BoardSnapException('TIMEOUT');
    } on http.ClientException {
      throw const BoardSnapException('NETWORK');
    } on FormatException {
      throw const BoardSnapException('INVALID_RESPONSE');
    } finally {
      client.close();
      if (identical(_active, client)) _active = null;
    }
  }

  Future<String> _send(
    http.Client client,
    Uri base,
    Uint8List bytes,
    String filename,
    ScanInputType inputType,
    BoardOrientation orientation,
  ) async {
    final endpoint = base.replace(
      path: '${base.path.replaceFirst(RegExp(r'/+$'), '')}/v1/recognize',
    );
    final request = http.MultipartRequest('POST', endpoint)
      ..followRedirects = false
      ..headers['Accept'] = 'application/json'
      ..fields['inputType'] = inputType.name
      ..fields['orientation'] = orientation.wireValue
      ..files.add(
        http.MultipartFile.fromBytes('image', bytes, filename: filename),
      );
    final streamed = await client.send(request);
    // Bound even malformed/proxy responses, and include body reading in timeout.
    final body = BytesBuilder();
    await for (final chunk in streamed.stream) {
      if (body.length + chunk.length > 64 * 1024) {
        throw const BoardSnapException('INVALID_RESPONSE');
      }
      body.add(chunk);
    }
    final dynamic data = jsonDecode(utf8.decode(body.takeBytes()));
    if (streamed.statusCode == 200 &&
        data is Map<String, dynamic> &&
        data.length == 1 &&
        data['piecePlacement'] is String) {
      final placement = data['piecePlacement'] as String;
      if (encodePiecePlacement(decodePiecePlacement(placement)) != placement) {
        throw const FormatException('Noncanonical placement.');
      }
      return placement;
    }
    if (streamed.statusCode != 200 &&
        data is Map<String, dynamic> &&
        data['error'] is Map &&
        data['error']['code'] is String) {
      throw BoardSnapException(data['error']['code'] as String);
    }
    throw const BoardSnapException('INVALID_RESPONSE');
  }

  void close() {
    _active?.close();
  }
}
