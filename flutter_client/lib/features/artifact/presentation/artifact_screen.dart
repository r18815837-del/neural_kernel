import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/error/error_handler.dart';
import '../../../core/utils/date_utils.dart';
import '../../../core/utils/file_utils.dart';
import '../../../shared/widgets/nk_app_bar.dart';
import '../../../core/error/error_widgets.dart';
import '../../../shared/widgets/nk_loading.dart';
import '../artifact_providers.dart';
import '../data/download_service.dart';
import 'download_button.dart';

class ArtifactScreen extends ConsumerWidget {
  final String projectId;

  const ArtifactScreen({super.key, required this.projectId});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final metaAsync = ref.watch(artifactMetaProvider(projectId));
    final downloadAsync = ref.watch(downloadStateProvider(projectId));

    return Scaffold(
      appBar: const NKAppBar(title: 'Artifact', showBack: true),
      body: metaAsync.when(
        loading: () => const NKLoading(message: 'Loading artifact info...'),
        error: (err, _) => FullScreenError(error: handleError(err)),
        data: (meta) => Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                meta.artifactName,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              if (meta.version != null)
                Text(
                  'Version ${meta.version}',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              const SizedBox(height: 12),

              _row(context, 'Project', meta.projectName),
              _row(context, 'Size', FileUtils.formatBytes(meta.sizeBytes)),
              _row(context, 'Format', meta.format.toUpperCase()),
              if (meta.createdAt != null)
                _row(context, 'Created', NKDateUtils.fullTimestamp(meta.createdAt)),

              const SizedBox(height: 24),

              Center(
                child: downloadAsync.when(
                  loading: () => const DownloadButton(
                    state: DownloadPreparing(),
                    onDownload: _noop,
                  ),
                  error: (err, _) => DownloadButton(
                    state: DownloadFailed(err.toString()),
                    onDownload: () =>
                        ref.invalidate(downloadStateProvider(projectId)),
                  ),
                  data: (state) => DownloadButton(
                    state: state,
                    onDownload: () =>
                        ref.invalidate(downloadStateProvider(projectId)),
                    onOpen: state is DownloadCompleted
                        ? () {
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content: Text('Saved: ${state.filePath}'),
                              ),
                            );
                          }
                        : null,
                    // TODO: Share functionality (share_plus)
                    onShare: null,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _row(BuildContext context, String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        children: [
          SizedBox(
            width: 80,
            child: Text(
              label,
              style: Theme.of(context)
                  .textTheme
                  .bodySmall
                  ?.copyWith(color: Colors.grey.shade600),
            ),
          ),
          Expanded(
            child: Text(value, style: Theme.of(context).textTheme.bodyMedium),
          ),
        ],
      ),
    );
  }

  static void _noop() {}
}
