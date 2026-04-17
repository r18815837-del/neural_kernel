import 'package:flutter/material.dart';

/// Color mapping for project statuses.
class StatusColors {
  StatusColors._();

  static Color forStatus(String status) => switch (status) {
        'pending' => Colors.orange,
        'in_progress' => Colors.blue,
        'completed' => Colors.green,
        'failed' => Colors.red,
        'archived' => Colors.grey,
        'cancelled' => Colors.grey.shade600,
        _ => Colors.grey,
      };

  static Color forStatusBackground(String status) =>
      forStatus(status).withValues(alpha: 0.12);
}
