import 'dart:io';

import 'package:path_provider/path_provider.dart';

/// Helpers for download paths and file size formatting.
class FileUtils {
  FileUtils._();

  /// Get platform-appropriate download directory.
  static Future<String> get downloadDir async {
    if (Platform.isAndroid) {
      final dir = await getExternalStorageDirectory();
      return dir?.path ?? (await getApplicationDocumentsDirectory()).path;
    }
    return (await getApplicationDocumentsDirectory()).path;
  }

  /// Human-readable file size.
  static String formatBytes(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    if (bytes < 1024 * 1024 * 1024) {
      return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
    }
    return '${(bytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB';
  }

  /// Check if file already downloaded.
  static Future<bool> fileExists(String path) => File(path).exists();
}
