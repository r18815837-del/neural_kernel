import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../core/auth/auth_provider.dart';
import '../../../shared/constants/strings.dart';
import '../../../shared/widgets/nk_app_bar.dart';

/// Settings screen — Phase 2, but included as a stub for navigation.
class SettingsScreen extends ConsumerWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final authState = ref.watch(authStateProvider);

    return Scaffold(
      appBar: const NKAppBar(title: Strings.settings, showBack: true),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Connection info
          authState.whenOrNull(
                data: (a) => a.credentials != null
                    ? Card(
                        child: Padding(
                          padding: const EdgeInsets.all(16),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Connected to',
                                style: Theme.of(context).textTheme.labelMedium,
                              ),
                              const SizedBox(height: 4),
                              Text(
                                a.credentials!.baseUrl,
                                style: Theme.of(context).textTheme.bodyLarge,
                              ),
                              const SizedBox(height: 4),
                              Text(
                                'Auth: ${a.credentials!.mode.label}',
                                style: Theme.of(context).textTheme.bodySmall,
                              ),
                            ],
                          ),
                        ),
                      )
                    : null,
              ) ??
              const SizedBox.shrink(),

          const SizedBox(height: 24),

          // Disconnect
          OutlinedButton.icon(
            onPressed: () async {
              await ref.read(authStateProvider.notifier).logout();
              if (context.mounted) context.go('/connect');
            },
            icon: const Icon(Icons.logout),
            label: const Text(Strings.disconnect),
            style: OutlinedButton.styleFrom(
              foregroundColor: Colors.red,
            ),
          ),

          const SizedBox(height: 32),

          // App info
          Center(
            child: Text(
              'Neural Kernel Client v0.1.0',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: Colors.grey.shade400,
                  ),
            ),
          ),
        ],
      ),
    );
  }
}
