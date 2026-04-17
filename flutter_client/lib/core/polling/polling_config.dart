/// Polling intervals per project status.
class PollingConfig {
  PollingConfig._();

  /// How often to poll based on backend project status.
  static Duration intervalFor(String status, int progressPercent) {
    return switch (status) {
      'pending' => const Duration(seconds: 2),
      'in_progress' when progressPercent < 50 => const Duration(seconds: 3),
      'in_progress' => const Duration(seconds: 5),
      _ => Duration.zero, // terminal — don't poll
    };
  }

  /// Whether this status is terminal (stop polling).
  static bool isTerminal(String status) {
    return const {'completed', 'failed', 'archived', 'cancelled'}
        .contains(status);
  }

  static const int maxRetries = 120;
  static const Duration maxInterval = Duration(seconds: 10);
  static const double backoffMultiplier = 1.5;
}
