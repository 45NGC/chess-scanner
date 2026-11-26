import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import '../../../app/l10n/app_localizations.dart';
import '../data/boardsnap_client.dart';
import '../data/scan_image_picker.dart';
import '../domain/piece_placement.dart';

class ScanPage extends StatefulWidget {
    const ScanPage({
        super.key,
        this.client,
        this.imagePicker,
        this.initialSource = ImageSource.gallery,
        this.openPickerOnStart = false,
    });
    final BoardSnapClient? client;
    final ScanImagePicker? imagePicker;
    final ImageSource initialSource;
    final bool openPickerOnStart;
    @override
    State<ScanPage> createState() => _ScanPageState();
}

class _ScanPageState extends State<ScanPage> {
    late final BoardSnapClient _client;
    late final ScanImagePicker _imagePicker;
    ImageSource? _source;
    Uint8List? _bytes;
    String _filename = '';
    ScanInputType? _inputType;
    BoardOrientation _orientation = BoardOrientation.whiteBottom;
    List<String>? _squares;
    String? _error;
    bool _busy = true;
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
        _imagePicker = widget.imagePicker ?? ScanImagePicker();
        if (widget.initialSource == ImageSource.camera) {
            _inputType = ScanInputType.photo;
        }
        WidgetsBinding.instance.addPostFrameCallback((_) => _initialize());
    }

    Future<void> _initialize() async {
        if (!mounted) return;
        var recovered = false;
        try {
            final image = await _imagePicker.recover();
            if (!mounted) return;
            if (image != null) {
                recovered = true;
                await _load(image.file, image.source);
            }
        }
        on BoardSnapException catch (error) {
            recovered = true;
            if (mounted) setState(() => _error = error.code);
        }
        on PlatformException catch (error) {
            recovered = true;
            if (mounted) setState(() => _error = _pickerError(error));
        }
        catch (_) {
            recovered = true;
            if (mounted) setState(() => _error = 'PICKER');
        }
        finally {
            if (mounted) setState(() => _busy = false);
        }
        if (mounted && !recovered &&
            (widget.openPickerOnStart || widget.initialSource == ImageSource.camera)) {
            await _choose(widget.initialSource);
        }
    }

    String _pickerError(PlatformException error) => switch (error.code) {
        'camera_access_denied' ||
        'camera_access_denied_without_prompt' ||
        'camera_access_restricted' => 'CAMERA_PERMISSION',
        'no_available_camera' || 'invalid_source' => 'CAMERA_UNAVAILABLE',
        _ => 'PICKER',
    };

    @override
    void dispose() {
        _client.close();
        super.dispose();
    }

    Future<void> _load(XFile file, ImageSource source) async {
        if (await file.length() > BoardSnapClient.maxImageBytes) {
            throw const BoardSnapException('PAYLOAD_TOO_LARGE');
        }
        final bytes = await file.readAsBytes();
        if (bytes.isEmpty) throw const BoardSnapException('INVALID_IMAGE');
        if (!mounted) return;
        setState(() {
            _bytes = bytes;
            _filename = file.name;
            _source = source;
            _inputType = source == ImageSource.camera
                ? ScanInputType.photo
                : null;
            _squares = null;
            _error = null;
        }
        );
    }

    Future<void> _choose(ImageSource source) async {
        if (_busy) return;
        if (source == ImageSource.camera && !_imagePicker.supportsCamera) {
            setState(() => _error = 'CAMERA_UNAVAILABLE');
            return;
        }
        setState(() {
            _busy = true;
            _error = null;
        }
        );
        try {
            final file = await _imagePicker.pick(source);
            if (file != null) await _load(file, source);
        }
        on BoardSnapException catch (error) {
            if (mounted) setState(() => _error = error.code);
        }
        on PlatformException catch (error) {
            if (mounted) setState(() => _error = _pickerError(error));
        }
        catch (_) {
            if (mounted) setState(() => _error = 'PICKER');
        }
        finally {
            if (mounted) setState(() => _busy = false);
        }
    }

    Future<void> _recognize() async {
        final bytes = _bytes;
        final inputType = _inputType;
        if (bytes == null || inputType == null || _busy) return;
        setState(() {
            _busy = true;
            _error = null;
        }
        );
        try {
            final placement = await _client.recognize(
                bytes: bytes,
                filename: _filename,
                inputType: inputType,
                orientation: _orientation,
            );
            if (mounted) setState(() => _squares = decodePiecePlacement(placement));
        }
        on BoardSnapException catch (error) {
            if (mounted) setState(() => _error = error.code);
        }
        catch (_) {
            if (mounted) setState(() => _error = 'NETWORK');
        }
        finally {
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
        'CAMERA_PERMISSION' => l.scanCameraPermission,
        'CAMERA_UNAVAILABLE' => l.scanCameraUnavailable,
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
                                    onPressed: _busy ? null : () => _choose(ImageSource.gallery),
                                    icon: const Icon(Icons.image_outlined),
                                    label: Text(l.homeImportAction),
                                ),
                                const SizedBox(height: 8),
                                OutlinedButton.icon(
                                    onPressed: _busy || !_imagePicker.supportsCamera
                                        ? null
                                        : () => _choose(ImageSource.camera),
                                    icon: const Icon(Icons.camera_alt_outlined),
                                    label: Text(
                                        _source == ImageSource.camera && _bytes != null
                                            ? l.scanRetake
                                            : l.homeCameraAction,
                                    ),
                                ),
                                if (!_imagePicker.supportsCamera) Text(l.scanCameraUnavailable),
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
                                    key: ValueKey(_inputType),
                                    initialValue: _inputType,
                                    hint: Text(l.scanChooseInputType),
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
                                    onPressed: _busy || _bytes == null || _inputType == null
                                        ? null : _recognize,
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
