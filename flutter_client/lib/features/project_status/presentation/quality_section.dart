import 'package:flutter/material.dart';

import '../domain/quality_score.dart';

/// Quality metrics display: scaffold_valid, execution_ready, consistency, overall.
class QualitySection extends StatelessWidget {
  final QualityScore quality;

  const QualitySection({super.key, required this.quality});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Quality', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        Wrap(
          spacing: 12,
          runSpacing: 8,
          children: [
            if (quality.scaffoldValid != null)
              _metric(context, 'Scaffold', quality.scaffoldValid!),
            if (quality.executionReady != null)
              _metric(context, 'Execution', quality.executionReady!),
            if (quality.consistencyOk != null)
              _metric(context, 'Consistency', quality.consistencyOk!),
            if (quality.overallPercent != null)
              _scoreBadge(context, quality.overallPercent!),
          ],
        ),
      ],
    );
  }

  Widget _metric(BuildContext context, String label, bool ok) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(
          ok ? Icons.check_circle : Icons.cancel,
          size: 18,
          color: ok ? Colors.green : Colors.red,
        ),
        const SizedBox(width: 4),
        Text(label, style: Theme.of(context).textTheme.bodySmall),
      ],
    );
  }

  Widget _scoreBadge(BuildContext context, int percent) {
    final color = percent >= 80
        ? Colors.green
        : percent >= 50
            ? Colors.orange
            : Colors.red;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        'Score: $percent%',
        style: TextStyle(
          color: color,
          fontWeight: FontWeight.w600,
          fontSize: 12,
        ),
      ),
    );
  }
}
