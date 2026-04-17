import 'package:flutter/material.dart';

import '../../../core/utils/date_utils.dart';
import '../../../shared/widgets/nk_chip.dart';
import '../../../shared/widgets/nk_status_badge.dart';
import '../domain/project.dart';

/// Card displaying a single project in the list.
class ProjectCard extends StatelessWidget {
  final ProjectListItem project;
  final VoidCallback onTap;

  const ProjectCard({
    super.key,
    required this.project,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Top row: name + status badge
              Row(
                children: [
                  Expanded(
                    child: Text(
                      project.name.isEmpty ? project.id : project.name,
                      style: Theme.of(context).textTheme.titleSmall?.copyWith(
                            fontWeight: FontWeight.w600,
                          ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  const SizedBox(width: 8),
                  NKStatusBadge(status: project.status),
                ],
              ),
              const SizedBox(height: 8),

              // Bottom row: time + features + artifact
              Row(
                children: [
                  Icon(
                    Icons.access_time,
                    size: 14,
                    color: Colors.grey.shade500,
                  ),
                  const SizedBox(width: 4),
                  Text(
                    NKDateUtils.relativeTime(project.createdAt),
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: Colors.grey.shade600,
                        ),
                  ),
                  const Spacer(),
                  if (project.artifactAvailable)
                    Icon(
                      Icons.download_done_rounded,
                      size: 18,
                      color: Colors.green.shade600,
                    ),
                ],
              ),

              // Feature chips (max 3)
              if (project.features.isNotEmpty) ...[
                const SizedBox(height: 8),
                Wrap(
                  spacing: 6,
                  runSpacing: 4,
                  children: project.features
                      .take(3)
                      .map((f) => NKChip(label: f))
                      .toList(),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
