import 'package:flutter/material.dart';

import '../../core/utils/extensions.dart';
import '../constants/status_colors.dart';

/// Colored badge displaying project status.
class NKStatusBadge extends StatelessWidget {
  final String status;

  const NKStatusBadge({super.key, required this.status});

  @override
  Widget build(BuildContext context) {
    final color = StatusColors.forStatus(status);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withValues(alpha: 0.4)),
      ),
      child: Text(
        _label(status),
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }

  String _label(String s) => s.replaceAll('_', ' ').capitalized;
}
