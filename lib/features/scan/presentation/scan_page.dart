import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import '../../../app/l10n/app_localizations.dart';
import '../data/boardsnap_client.dart';
import '../domain/piece_placement.dart';

class ScanPage extends StatefulWidget {
  const ScanPage({super.key, this.client, this.pickImage});
  final BoardSnapClient? client;
  final Future<XFile?> Function()? pickImage;
  @override
  State<ScanPage> createState() => _ScanPageState();
}

class _ScanPageState extends State<ScanPage> {
  late final BoardSnapClient _client;
  Uint8List? _bytes;
  String _filename = '';
  ScanInputType _inputType = ScanInputType.digital;
  BoardOrientation _orientation = BoardOrientation.whiteBottom;
  List<String>? _squares;
  String? _error;
  bool _busy = false;
  static const _glyphs = {
    'K': '♔',
    'Q': '♕',
    'R': '♖',
    'B': '♗',
    'N': '♘',
    'P': '♙',
    'k': '♚',
    'q': '♛',
    'r': '♜',
    'b': '♝',
    'n': '♞',
    'p': '♟',
  };

  @override
  void initState() {
    super.initState();
    _client =
        widget.client ??
        BoardSnapClient(baseUrl: const String.fromEnvironment('BOARDSNAP_URL'));
    _recoverImage();
  }

  Future<void> _recoverImage() async {
    if (widget.pickImage != null ||
        kIsWeb ||
        defaultTargetPlatform != TargetPlatform.android) {
      return;
    }
    try {
      final response = await ImagePicker().retrieveLostData();
      if (!mounted || response.isEmpty) return;
      if (response.files?.isNotEmpty ?? false) {
        await _load(response.files!.first);
      } else if (response.exception != null) {
        setState(() => _error = 'PICKER');
      }
    } on BoardSnapException catch (error) {
      if (mounted) setState(() => _error = error.code);
    } catch (_) {
      if (mounted) setState(() => _error = 'PICKER');
    }
  }

  @override
  void dispose() {
    _client.close();
    super.dispose();
  }

  Future<void> _load(XFile file) async {
    if (await file.length() > BoardSnapClient.maxImageBytes) {
      throw const BoardSnapException('PAYLOAD_TOO_LARGE');
    }
    final bytes = await file.readAsBytes();
    if (bytes.isEmpty) throw const BoardSnapException('INVALID_IMAGE');
    if (!mounted) return;
    setState(() {
      _bytes = bytes;
      _filename = file.name;
      _squares = null;
      _error = null;
    });
  }

  Future<void> _choose() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final file =
          await (widget.pickImage?.call() ??
              ImagePicker().pickImage(source: ImageSource.gallery));
      if (file != null) await _load(file);
    } on BoardSnapException catch (error) {
      if (mounted) setState(() => _error = error.code);
    } catch (_) {
      if (mounted) setState(() => _error = 'PICKER');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _recognize() async {
    final bytes = _bytes;
    if (bytes == null || _busy) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final placement = await _client.recognize(
        bytes: bytes,
        filename: _filename,
        inputType: _inputType,
        orientation: _orientation,
      );
      if (mounted) setState(() => _squares = decodePiecePlacement(placement));
    } on BoardSnapException catch (error) {
      if (mounted) setState(() => _error = error.code);
    } catch (_) {
      if (mounted) setState(() => _error = 'NETWORK');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  String _message(AppLocalizations l, String code) => switch (code) {
    'CONFIGURATION' => l.scanConfigurationError,
    'BOARD_NOT_FOUND' => l.scanBoardNotFound,
    'INVALID_IMAGE' || 'UNSUPPORTED_IMAGE' => l.scanImageError,
    'PAYLOAD_TOO_LARGE' => l.scanSizeError,
    'TIMEOUT' => l.scanTimeout,
    'NETWORK' => l.scanNetworkError,
    'INVALID_RESPONSE' => l.scanResponseError,
    'PICKER' => l.scanPickerError,
    _ => l.scanServerError,
  };

  Future<void> _edit(int index) async {
    final l = AppLocalizations.of(context)!;
    final piece = await showModalBottomSheet<String>(
      context: context,
      builder: (context) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Wrap(
            spacing: 8,
            runSpacing: 8,
            children: ['', ..._glyphs.keys]
                .map(
                  (p) => OutlinedButton(
                    onPressed: () => Navigator.pop(context, p),
                    child: Text(
                      p.isEmpty ? l.scanEmpty : '${_glyphs[p]} $p',
                      style: const TextStyle(fontSize: 24),
                    ),
                  ),
                )
                .toList(),
          ),
        ),
      ),
    );
    if (piece != null && mounted) setState(() => _squares![index] = piece);
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context)!;
    return Scaffold(
      appBar: AppBar(title: Text(l.scanTitle)),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 600),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(l.scanUploadNotice),
                const SizedBox(height: 16),
                OutlinedButton.icon(
                  onPressed: _busy ? null : _choose,
                  icon: const Icon(Icons.image_outlined),
                  label: Text(l.homeImportAction),
                ),
                if (_bytes != null) ...[
                  const SizedBox(height: 12),
                  Image.memory(
                    _bytes!,
                    height: 220,
                    fit: BoxFit.contain,
                    errorBuilder: (_, _, _) => Text(l.scanImageError),
                  ),
                  const SizedBox(height: 16),
                ],
                DropdownButtonFormField<ScanInputType>(
                  initialValue: _inputType,
                  decoration: InputDecoration(labelText: l.scanInputType),
                  items: [
                    DropdownMenuItem(
                      value: ScanInputType.digital,
                      child: Text(l.scanDigital),
                    ),
                    DropdownMenuItem(
                      value: ScanInputType.photo,
                      child: Text(l.scanPhoto),
                    ),
                  ],
                  onChanged: _busy
                      ? null
                      : (value) => setState(() => _inputType = value!),
                ),
                const SizedBox(height: 16),
                DropdownButtonFormField<BoardOrientation>(
                  initialValue: _orientation,
                  decoration: InputDecoration(labelText: l.scanOrientation),
                  items: [
                    DropdownMenuItem(
                      value: BoardOrientation.whiteBottom,
                      child: Text(l.scanWhiteBottom),
                    ),
                    DropdownMenuItem(
                      value: BoardOrientation.blackBottom,
                      child: Text(l.scanBlackBottom),
                    ),
                  ],
                  onChanged: _busy
                      ? null
                      : (value) => setState(() => _orientation = value!),
                ),
                const SizedBox(height: 8),
                Text(l.scanOrientationHelp),
                const SizedBox(height: 16),
                FilledButton(
                  onPressed: _busy || _bytes == null ? null : _recognize,
                  child: Text(l.scanRecognize),
                ),
                if (_busy) ...[
                  const SizedBox(height: 16),
                  const LinearProgressIndicator(),
                  Text(l.scanWorking),
                ],
                if (_error != null)
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    child: Text(
                      _message(l, _error!),
                      key: const Key('scan-error'),
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.error,
                      ),
                    ),
                  ),
                if (_squares != null) ...[
                  const SizedBox(height: 24),
                  Text(l.scanReview),
                  const SizedBox(height: 12),
                  GridView.builder(
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    itemCount: 64,
                    gridDelegate:
                        const SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: 8,
                        ),
                    itemBuilder: (context, index) {
                      final piece = _squares![index];
                      final square =
                          '${String.fromCharCode(97 + index % 8)}${8 - index ~/ 8}';
                      return Semantics(
                        label: '$square ${piece.isEmpty ? l.scanEmpty : piece}',
                        button: true,
                        child: InkWell(
                          key: ValueKey('square-$index'),
                          onTap: _busy ? null : () => _edit(index),
                          child: Container(
                            color: (index ~/ 8 + index % 8).isEven
                                ? const Color(0xfff0d9b5)
                                : const Color(0xffb58863),
                            child: Stack(
                              children: [
                                Center(
                                  child: FittedBox(
                                    child: Text(
                                      _glyphs[piece] ?? '',
                                      style: const TextStyle(
                                        fontSize: 40,
                                        color: Colors.black,
                                      ),
                                    ),
                                  ),
                                ),
                                Positioned(
                                  left: 2,
                                  top: 1,
                                  child: Text(
                                    square,
                                    style: const TextStyle(
                                      fontSize: 9,
                                      color: Colors.black54,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                  const SizedBox(height: 16),
                  SelectableText(
                    encodePiecePlacement(_squares!),
                    key: const Key('piece-placement'),
                  ),
                  TextButton.icon(
                    onPressed: () async {
                      await Clipboard.setData(
                        ClipboardData(text: encodePiecePlacement(_squares!)),
                      );
                      if (context.mounted) {
                        ScaffoldMessenger.of(
                          context,
                        ).showSnackBar(SnackBar(content: Text(l.scanCopied)));
                      }
                    },
                    icon: const Icon(Icons.copy),
                    label: Text(l.scanCopy),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}
