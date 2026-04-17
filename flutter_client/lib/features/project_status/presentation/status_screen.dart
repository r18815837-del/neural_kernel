import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../core/error/error_handler.dart';
import '../../../core/utils/date_utils.dart';
import '../../../shared/widgets/nk_app_bar.dart';
import '../../../shared/widgets/nk_error_view.dart';
import '../../../shared/widgets/nk_loading.dart';
import '../../../shared/widgets/nk_status_badge.dart';
import '../project_status_providers.dart';
import 'artifact_section.dart';
import 'features_section.dart';
import 'pipeline_section.dart';
import 'progress_section.dart';
import 'quality_section.dart';
import 'tech_stack_section.dart';

class StatusScreen extends ConsumerWidget {
  final String projectId;

  const StatusScreen({super.key, required this.projectId});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final statusAsync = ref.watch(projectStatusStreamProvider(projectId));

    return Scaffold(
      appBar: NKAppBar(
        title: 'Project',
        showBack: true,
        actions: [
          // Future: lifecycle actions (archive, retry)
          statusAsync.whenOrNull(
                data: (s) => s.artifactAvailable
                    ? IconButton(
                        icon: const Icon(Icons.info_outline),
                        onPressed: () =>
                            context.push('/projects/$projectId/artifact'),
                      )
                    : null,
              ) ??
              const SizedBox.shrink(),
        ],
      ),
      body: statusAsync.when(
        loading: () => const NKLoading(message: 'Loading project...'),
        error: (err, _) => FullScreenError(
          error: handleError(err),
          onRetry: () => ref.invalidate(projectStatusStreamProvider(projectId)),
          onBack: () => context.go('/projects'),
        ),
        data: (status) => _buildBody(context, status, ref),
      ),
    );
  }

  Widget _buildBody(BuildContext context, status, WidgetRef ref) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Header: name + status badge
        Row(
          children: [
            Expanded(
              child: Text(
                status.name.isNotEmpty ? status.name : status.id,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
            ),
            NKStatusBadge(status: status.status),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          NKDateUtils.relativeTime(status.createdAt),
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Colors.grey.shade500,
              ),
        ),
        const SizedBox(height: 16),

        // Progress
        ProgressSection(
          percent: status.progressPercent,
          status: status.status,
        ),
        const SizedBox(height: 16),

        // Message
        if (status.message.isNotEmpty) ...[
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    Icons.message_outlined,
                    size: 20,
                    color: Colors.grey.shade500,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      status.message,
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
        ],

        // Error banner
        if (status.error != null) ...[
          Card(
            color: Colors.red.shade50,
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.error_outline, color: Colors.red.shade700),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      status.error!,
                      style: TextStyle(color: Colors.red.shade900),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
        ],

        // Quality
        if (status.quality != null) ...[
          QualitySection(quality: status.quality!),
          const SizedBox(height: 16),
        ],

        // Features
        FeaturesSection(features: status.features),
        if (status.features.isNotEmpty) const SizedBox(height: 16),

        // Tech stack
        if (status.techStack != null && !status.techStack!.isEmpty) ...[
          TechStackSection(techStack: status.techStack!),
          const SizedBox(height: 16),
        ],

        // Pipeline
        PipelineSection(status: status),
        if (status.agentCount > 0) const SizedBox(height: 16),

        // Artifact
        ArtifactSection(
          status: status,
          onDownload: () =>
              context.push('/projects/${status.id}/artifact'),
        ),
      ],
    );
  }
}
