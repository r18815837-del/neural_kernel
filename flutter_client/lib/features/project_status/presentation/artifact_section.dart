import 'package:flutter/material.dart';

import '../../../core/utils/file_utils.dart';
import '../../../shared/constants/strings.dart';
import '../domain/project_status.dart';

/// Artifact download button + metadata preview.
class ArtifactSection extends StatelessWidget {
  final ProjectStatus status;
  final VoidCallback? onDownload;

  const ArtifactSection({
    super.key,
    required this.status,
    this.onDownload,
  });

  @override
  Widget build(BuildContext context) {
    if (!status.artifactAvailable) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Artifact', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        Card(
          color: Colors.green.shade50,
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                Icon(Icons.archive_outlined, color: Colors.green.shade700),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        status.artifactName ?? 'Project artifact',
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                              fontWeight: FontWeight.w600,
                            ),
                      ),
                      if (status.artifactSizeBytes != null)
                        Text(
                          FileUtils.formatBytes(status.artifactSizeBytes!),
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                    ],
                  ),
                ),
                FilledButton.icon(
                  onPressed: onDownload,
                  icon: const Icon(Icons.download),
                  label: const Text(Strings.download),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
