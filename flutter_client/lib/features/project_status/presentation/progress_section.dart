import 'package:flutter/material.dart';

import '../../../shared/constants/status_colors.dart';

/// Progress bar + percentage display.
class ProgressSection extends StatelessWidget {
  final int percent;
  final String status;

  const ProgressSection({
    super.key,
    required this.percent,
    required this.status,
  });

  @override
  Widget build(BuildContext context) {
    final color = StatusColors.forStatus(status);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Progress',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            Text(
              '$percent%',
              style: Theme.of(context).textTheme.titleSmall?.copyWith(
                    color: color,
                    fontWeight: FontWeight.bold,
                  ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: LinearProgressIndicator(
            value: percent / 100,
            minHeight: 8,
            backgroundColor: Colors.grey.shade200,
            valueColor: AlwaysStoppedAnimation(color),
          ),
        ),
      ],
    );
  }
}
