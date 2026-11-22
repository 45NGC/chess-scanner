import 'package:flutter/foundation.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

class RecoveredScanImage {
  const RecoveredScanImage(this.file, this.source);
  final XFile file;
  final ImageSource source;
}

/// Native still-photo capture and gallery import. Never uploads an image.
class ScanImagePicker {
  ScanImagePicker({ImagePicker? picker}) : _picker = picker ?? ImagePicker();
  final ImagePicker _picker;
  static const pendingSourceKey = 'scan.pendingImageSource';

  static bool get nativeCameraSupported =>
      !kIsWeb &&
      (defaultTargetPlatform == TargetPlatform.android ||
          defaultTargetPlatform == TargetPlatform.iOS);

  bool get supportsCamera => nativeCameraSupported;
  bool get _needsRecovery =>
      !kIsWeb && defaultTargetPlatform == TargetPlatform.android;

  Future<XFile?> pick(ImageSource source) async {
    final preferences = _needsRecovery
        ? await SharedPreferences.getInstance()
        : null;
    await preferences?.setString(pendingSourceKey, source.name);
    try {
      return await _picker.pickImage(
        source: source,
        preferredCameraDevice: CameraDevice.rear,
        requestFullMetadata: false,
      );
    } finally {
      await preferences?.remove(pendingSourceKey);
    }
  }

  Future<RecoveredScanImage?> recover() async {
    if (!_needsRecovery) return null;
    final preferences = await SharedPreferences.getInstance();
    final source =
        preferences.getString(pendingSourceKey) == ImageSource.gallery.name
        ? ImageSource.gallery
        : ImageSource.camera;
    try {
      final response = await _picker.retrieveLostData();
      if (response.exception != null) throw response.exception!;
      if (response.files?.isNotEmpty ?? false) {
        return RecoveredScanImage(response.files!.first, source);
      }
      return null;
    } finally {
      await preferences.remove(pendingSourceKey);
    }
  }
}
