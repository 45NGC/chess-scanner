import 'package:chess_scanner/features/scan/data/scan_image_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

class NativePicker extends ImagePicker {
    Future<XFile?> Function(ImageSource)? choose;
    LostDataResponse? lost;
    bool? metadataRequested;
    CameraDevice? camera;
    @override
    Future<XFile?> pickImage({
        required ImageSource source,
        double? maxWidth,
        double? maxHeight,
        int? imageQuality,
        CameraDevice preferredCameraDevice = CameraDevice.rear,
        bool requestFullMetadata = true,
    }) async {
        metadataRequested = requestFullMetadata;
        camera = preferredCameraDevice;
        return await choose?.call(source);
    }

    @override
    Future<LostDataResponse> retrieveLostData() async =>
    lost ?? LostDataResponse.empty();
}

void main() {
    TestWidgetsFlutterBinding.ensureInitialized();
    setUp(() {
        SharedPreferences.setMockInitialValues({});
        debugDefaultTargetPlatformOverride = TargetPlatform.android;
    }
    );
    tearDown(() => debugDefaultTargetPlatformOverride = null);

    test(
        'camera source is saved during capture and cleared on cancellation',
        () async {
            final preferences = await SharedPreferences.getInstance();
            final native = NativePicker()
                ..choose = (source) async {
                expect(source, ImageSource.camera);
                expect(
                    preferences.getString(ScanImagePicker.pendingSourceKey),
                    'camera',
                );
                return null;
            };
            expect(
                await ScanImagePicker(picker: native).pick(ImageSource.camera),
                isNull,
            );
            expect(native.metadataRequested, isFalse);
            expect(native.camera, CameraDevice.rear);
            expect(
                preferences.containsKey(ScanImagePicker.pendingSourceKey),
                isFalse,
            );
        },
    );

    test(
        'camera denial preserves the plugin error and clears pending state',
        () async {
            final native = NativePicker()
                ..choose = (_) async =>
            throw PlatformException(code: 'camera_access_denied');
            await expectLater(
                ScanImagePicker(picker: native).pick(ImageSource.camera),
                throwsA(
                    isA<PlatformException>().having(
                        (e) => e.code,
                        'code',
                        'camera_access_denied',
                    ),
                ),
            );
            expect(
                (await SharedPreferences.getInstance()).containsKey(
                    ScanImagePicker.pendingSourceKey,
                ),
                isFalse,
            );
        },
    );

    for (final source in ImageSource.values) {
        test('Android restores $source without changing its source', () async {
            final preferences = await SharedPreferences.getInstance();
            await preferences.setString(
                ScanImagePicker.pendingSourceKey,
                source.name,
            );
            final file = XFile.fromData(
                Uint8List.fromList([1]),
                name: 'recovered.jpg',
            );
            final native = NativePicker()..lost = LostDataResponse(files: [file]);
            final result = await ScanImagePicker(picker: native).recover();
            expect(result!.source, source);
            expect(result.file, same(file));
            expect(
                preferences.containsKey(ScanImagePicker.pendingSourceKey),
                isFalse,
            );
        }
        );
    }
}
