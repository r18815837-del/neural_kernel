import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../core/auth/auth_provider.dart';
import '../features/artifact/presentation/artifact_screen.dart';
import '../features/ask/presentation/ask_screen.dart';
import '../features/code/presentation/code_screen.dart';
import '../features/auth/presentation/connect_screen.dart';
import '../features/create_project/presentation/create_screen.dart';
import '../features/project_status/presentation/status_screen.dart';
import '../features/projects/presentation/projects_screen.dart';
import '../features/settings/presentation/settings_screen.dart';

/// GoRouter with auth-aware redirects.
final routerProvider = Provider<GoRouter>((ref) {
  // Rebuild router when auth changes.
  final authAsync = ref.watch(authStateProvider);

  return GoRouter(
    initialLocation: '/connect',
    debugLogDiagnostics: true,
    redirect: (context, state) {
      final authData = authAsync.valueOrNull;
      final isAuthenticated = authData?.isAuthenticated ?? false;
      final isOnConnect = state.matchedLocation == '/connect';

      if (!isAuthenticated && !isOnConnect) return '/connect';
      if (isAuthenticated && isOnConnect) return '/projects';
      return null;
    },
    routes: [
      GoRoute(
        path: '/connect',
        builder: (_, __) => const ConnectScreen(),
      ),
      GoRoute(
        path: '/projects',
        builder: (_, __) => const ProjectsScreen(),
      ),
      GoRoute(
        path: '/projects/create',
        builder: (_, __) => const CreateProjectScreen(),
      ),
      GoRoute(
        path: '/projects/:id',
        builder: (_, state) => StatusScreen(
          projectId: state.pathParameters['id']!,
        ),
      ),
      GoRoute(
        path: '/projects/:id/artifact',
        builder: (_, state) => ArtifactScreen(
          projectId: state.pathParameters['id']!,
        ),
      ),
      GoRoute(
        path: '/ask',
        builder: (_, __) => const AskScreen(),
      ),
      GoRoute(
        path: '/code',
        builder: (_, __) => const CodeScreen(),
      ),
      GoRoute(
        path: '/settings',
        builder: (_, __) => const SettingsScreen(),
      ),
    ],
  );
});
