import 'package:flutter/material.dart';

import 'error_models.dart';

/// A banner shown at the top of a screen for non-critical errors.
class ErrorBanner extends StatelessWidget {
  final AppError error;
  final VoidCallback? onRetry;

  const ErrorBanner({super.key, required this.error, this.onRetry});

  @override
  Widget build(BuildContext context) {
    return MaterialBanner(
      backgroundColor: Colors.red.shade50,
      content: Text(
        error.userMessage,
        style: TextStyle(color: Colors.red.shade900),
      ),
      actions: [
        if (error.retryable && onRetry != null)
          TextButton(
            onPressed: onRetry,
            child: const Text('Retry'),
          ),
        TextButton(
          onPressed: () {
            ScaffoldMessenger.of(context).hideCurrentMaterialBanner();
          },
          child: const Text('Dismiss'),
        ),
      ],
    );
  }
}

/// Full-screen error with optional retry.
class FullScreenError extends StatelessWidget {
  final AppError error;
  final VoidCallback? onRetry;
  final VoidCallback? onBack;

  const FullScreenError({
    super.key,
    required this.error,
    this.onRetry,
    this.onBack,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              _icon(error.type),
              size: 64,
              color: Colors.grey.shade400,
            ),
            const SizedBox(height: 16),
            Text(
              error.userMessage,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 24),
            if (error.retryable && onRetry != null)
              FilledButton.icon(
                onPressed: onRetry,
                icon: const Icon(Icons.refresh),
                label: const Text('Try again'),
              ),
            if (onBack != null) ...[
              const SizedBox(height: 12),
              OutlinedButton(
                onPressed: onBack,
                child: const Text('Go back'),
              ),
            ],
          ],
        ),
      ),
    );
  }

  IconData _icon(AppErrorType type) => switch (type) {
        AppErrorType.network => Icons.wifi_off_rounded,
        AppErrorType.notFound => Icons.search_off_rounded,
        AppErrorType.unauthorized => Icons.lock_outline_rounded,
        _ => Icons.error_outline_rounded,
      };
}
