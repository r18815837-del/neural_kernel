import 'package:flutter/material.dart';

import '../../../core/auth/auth_models.dart';
import '../../../shared/constants/strings.dart';

/// The connect form widget — extracted for testability.
class ConnectForm extends StatelessWidget {
  final TextEditingController urlController;
  final TextEditingController secretController;
  final AuthMode mode;
  final bool loading;
  final String? error;
  final ValueChanged<AuthMode> onModeChanged;
  final VoidCallback onConnect;

  const ConnectForm({
    super.key,
    required this.urlController,
    required this.secretController,
    required this.mode,
    required this.loading,
    required this.error,
    required this.onModeChanged,
    required this.onConnect,
  });

  @override
  Widget build(BuildContext context) {
    return ConstrainedBox(
      constraints: const BoxConstraints(maxWidth: 400),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Logo / title
          Icon(
            Icons.hub_rounded,
            size: 64,
            color: Theme.of(context).colorScheme.primary,
          ),
          const SizedBox(height: 12),
          Text(
            Strings.appName,
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
          ),
          const SizedBox(height: 32),

          // Server URL
          TextField(
            controller: urlController,
            decoration: const InputDecoration(
              labelText: Strings.enterBaseUrl,
              hintText: Strings.enterBaseUrlHint,
              prefixIcon: Icon(Icons.dns_outlined),
              border: OutlineInputBorder(),
            ),
            keyboardType: TextInputType.url,
            textInputAction: TextInputAction.next,
          ),
          const SizedBox(height: 16),

          // Auth mode toggle
          SegmentedButton<AuthMode>(
            segments: [
              for (final m in AuthMode.values)
                ButtonSegment(value: m, label: Text(m.label)),
            ],
            selected: {mode},
            onSelectionChanged: (s) => onModeChanged(s.first),
          ),
          if (mode != AuthMode.none) ...[
            const SizedBox(height: 16),

            // Secret field
            TextField(
              controller: secretController,
              decoration: InputDecoration(
                labelText: Strings.enterSecret,
                prefixIcon: const Icon(Icons.key_outlined),
                border: const OutlineInputBorder(),
                suffixIcon: mode == AuthMode.bearer
                    ? const Tooltip(
                        message: 'JWT token from your identity provider',
                        child: Icon(Icons.info_outline),
                      )
                    : null,
              ),
              obscureText: true,
              textInputAction: TextInputAction.done,
              onSubmitted: (_) => onConnect(),
            ),
          ],
          const SizedBox(height: 8),

          // Error
          if (error != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(
                error!,
                style: TextStyle(color: Colors.red.shade700, fontSize: 13),
              ),
            ),

          const SizedBox(height: 16),

          // Connect button
          FilledButton(
            onPressed: loading ? null : onConnect,
            child: loading
                ? const SizedBox(
                    height: 20,
                    width: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Colors.white,
                    ),
                  )
                : const Text(Strings.connect),
          ),
        ],
      ),
    );
  }
}
