import 'dart:async';

import 'package:dio/dio.dart';

import '../../../core/utils/file_utils.dart';
import 'artifact_api.dart';

/// Sealed download state.
sealed class DownloadState {
  const DownloadState();
}

class DownloadIdle extends DownloadState {
  const DownloadIdle();
}

class DownloadPreparing extends DownloadState {
  const DownloadPreparing();
}

class Downloading extends DownloadState {
  final double progress; // 0.0 - 1.0
  final int bytesReceived;
  final int totalBytes;

  const Downloading({
    required this.progress,
    required this.bytesReceived,
    required this.totalBytes,
  });
}

class DownloadCompleted extends DownloadState {
  final String filePath;

  const DownloadCompleted(this.filePath);
}

class DownloadFailed extends DownloadState {
  final String error;
  final bool retryable;

  const DownloadFailed(this.error, {this.retryable = true});
}

/// Service managing artifact download with progress tracking.
class DownloadService {
  final ArtifactApi _api;
  CancelToken? _cancelToken;

  DownloadService(this._api);

  /// Download artifact, yielding [DownloadState] events.
  Stream<DownloadState> download(String projectId) async* {
    yield const DownloadPreparing();

    try {
      // 1. Fetch metadata.
      final meta = await _api.getMetadata(projectId);

      // 2. Determine save path.
      final dir = await FileUtils.downloadDir;
      final savePath = '$dir/${meta.artifactName}';

      // Check if already downloaded.
      if (await FileUtils.fileExists(savePath)) {
        yield DownloadCompleted(savePath);
        return;
      }

      // 3. Download with progress.
      _cancelToken = CancelToken();
      final controller = StreamController<DownloadState>();

      _api.downloadArtifact(
        projectId: projectId,
        savePath: savePath,
        cancelToken: _cancelToken,
        onProgress: (received, total) {
          if (total > 0) {
            controller.add(
              Downloading(
                progress: received / total,
                bytesReceived: received,
                totalBytes: total,
              ),
            );
          }
        },
      ).then((_) {
        controller.add(DownloadCompleted(savePath));
        controller.close();
      }).catchError((e) {
        controller.add(DownloadFailed(e.toString()));
        controller.close();
      });

      yield* controller.stream;
    } catch (e) {
      yield DownloadFailed(e.toString());
    }
  }

  void cancel() {
    _cancelToken?.cancel('User cancelled');
  }
}
