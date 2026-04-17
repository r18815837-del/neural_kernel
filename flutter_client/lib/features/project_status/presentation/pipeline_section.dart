import 'package:flutter/material.dart';

import '../domain/project_status.dart';

/// Agent pipeline summary: agent count, success/fail, LLM badge.
class PipelineSection extends StatelessWidget {
  final ProjectStatus status;

  const PipelineSection({super.key, required this.status});

  @override
  Widget build(BuildContext context) {
    if (status.agentCount == 0) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Pipeline', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        Row(
          children: [
            _stat(
              context, Icons.smart_toy_outlined,
              '${status.agentCount}', 'agents',
            ),
            const SizedBox(width: 16),
            _stat(
              context, Icons.check_circle_outline,
              '${status.successfulAgentCount}', 'passed',
              color: Colors.green,
            ),
            if (status.failedAgentCount > 0) ...[
              const SizedBox(width: 16),
              _stat(
                context, Icons.error_outline,
                '${status.failedAgentCount}', 'failed',
                color: Colors.red,
              ),
            ],
            const Spacer(),
            if (status.llmUsed)
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: Colors.purple.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Text(
                  'LLM',
                  style: TextStyle(
                    color: Colors.purple,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
          ],
        ),
      ],
    );
  }

  Widget _stat(
    BuildContext context,
    IconData icon,
    String value,
    String label, {
    Color? color,
  }) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 16, color: color ?? Colors.grey.shade600),
        const SizedBox(width: 4),
        Text(
          value,
          style: TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 14,
            color: color,
          ),
        ),
        const SizedBox(width: 2),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Colors.grey.shade500,
              ),
        ),
      ],
    );
  }
}
