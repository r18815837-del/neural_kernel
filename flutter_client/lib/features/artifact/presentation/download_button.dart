import 'package:flutter/material.dart';

import '../../../core/utils/file_utils.dart';
import '../data/download_service.dart';

/// Download button with built-in progress indicator.
class DownloadButton extends StatelessWidget {
  final DownloadState state;
  final VoidCallback onDownload;
  final VoidCallback? onOpen;
  final VoidCallback? onShare;

  const DownloadButton({
    super.key,
    required this.state,
    required this.onDownload,
    this.onOpen,
    this.onShare,
  });

  @override
  Widget build(BuildContext context) {
    return switch (state) {
      DownloadIdle() => FilledButton.icon(
          onPressed: onDownload,
          icon: const Icon(Icons.download),
          label: const Text('Download'),
        ),
      DownloadPreparing() => FilledButton.icon(
          onPressed: null,
          icon: const SizedBox(
            width: 18,
            height: 18,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          label: const Text('Preparing...'),
        ),
      Downloading(
        progress: final p,
        bytesReceived: final r,
        totalBytes: final t,
      ) =>
        Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: double.infinity,
              child: LinearProgressIndicator(value: p),
            ),
            const SizedBox(height: 4),
            Text(
              '${FileUtils.formatBytes(r)} / ${FileUtils.formatBytes(t)}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      DownloadCompleted() => Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            FilledButton.icon(
              onPressed: onOpen,
              icon: const Icon(Icons.folder_open),
              label: const Text('Open'),
            ),
            const SizedBox(width: 8),
            OutlinedButton.icon(
              onPressed: onShare,
              icon: const Icon(Icons.share),
              label: const Text('Share'),
            ),
          ],
        ),
      DownloadFailed(error: final e, retryable: final canRetry) => Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              e,
              style: TextStyle(color: Colors.red.shade700, fontSize: 13),
            ),
            if (canRetry) ...[
              const SizedBox(height: 8),
              OutlinedButton.icon(
                onPressed: onDownload,
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
              ),
            ],
          ],
        ),
    };
  }
}
