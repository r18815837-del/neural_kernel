import 'dart:async';

import 'polling_config.dart';

/// Generic poll-until-done stream controller.
///
/// Usage:
/// ```dart
/// final stream = PollingController.poll(
///   fetcher: () => api.getStatus(projectId),
///   isDone: (s) => PollingConfig.isTerminal(s.status),
///   interval: (s) => PollingConfig.intervalFor(s.status, s.progressPercent),
/// );
/// ```
class PollingController {
  PollingController._();

  /// Creates a stream that polls [fetcher] until [isDone] returns true.
  ///
  /// On network errors, backs off exponentially but keeps polling.
  /// On 404 / auth errors, rethrows to stop the stream.
  static Stream<T> poll<T>({
    required Future<T> Function() fetcher,
    required bool Function(T) isDone,
    required Duration Function(T) interval,
    int maxRetries = PollingConfig.maxRetries,
    Duration maxInterval = PollingConfig.maxInterval,
    double backoffMultiplier = PollingConfig.backoffMultiplier,
  }) async* {
    var retries = 0;
    var currentBackoff = Duration.zero;

    while (retries < maxRetries) {
      try {
        final result = await fetcher();
        currentBackoff = Duration.zero; // reset on success
        yield result;

        if (isDone(result)) break;

        final wait = interval(result);
        if (wait == Duration.zero) break;
        await Future.delayed(wait);
        retries++;
      } catch (e) {
        // Exponential backoff for transient errors.
        if (currentBackoff == Duration.zero) {
          currentBackoff = const Duration(seconds: 2);
        } else {
          currentBackoff = Duration(
            milliseconds:
                (currentBackoff.inMilliseconds * backoffMultiplier).toInt(),
          );
          if (currentBackoff > maxInterval) currentBackoff = maxInterval;
        }
        await Future.delayed(currentBackoff);
        retries++;

        // Rethrow non-transient errors.
        if (_isNonTransient(e)) rethrow;
      }
    }
  }

  static bool _isNonTransient(Object e) {
    final msg = e.toString().toLowerCase();
    return msg.contains('unauthorized') ||
        msg.contains('not found') ||
        msg.contains('forbidden');
  }
}
