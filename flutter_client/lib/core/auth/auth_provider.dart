import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/api_client.dart';
import 'auth_models.dart';
import 'auth_service.dart';

/// Singleton auth service.
final authServiceProvider = Provider<AuthService>((ref) => AuthService());

/// Global auth state — drives router guards and API client config.
final authStateProvider =
    AsyncNotifierProvider<AuthStateNotifier, AuthState>(
  AuthStateNotifier.new,
);

class AuthStateNotifier extends AsyncNotifier<AuthState> {
  @override
  Future<AuthState> build() async {
    final service = ref.read(authServiceProvider);
    final creds = await service.loadCredentials();
    if (creds != null && creds.isValid) {
      // Set base URL for API client.
      ref.read(baseUrlProvider.notifier).state = creds.baseUrl;
      return AuthState.authenticated(creds);
    }
    return const AuthState.unauthenticated();
  }

  /// Called from ConnectScreen after validation passes.
  Future<void> login(AuthCredentials credentials) async {
    final service = ref.read(authServiceProvider);
    await service.saveCredentials(credentials);
    // Update base URL so Dio picks it up.
    ref.read(baseUrlProvider.notifier).state = credentials.baseUrl;
    state = AsyncData(AuthState.authenticated(credentials));
  }

  /// Called from Settings or on 401.
  Future<void> logout() async {
    final service = ref.read(authServiceProvider);
    await service.clearCredentials();
    ref.read(baseUrlProvider.notifier).state = '';
    state = const AsyncData(AuthState.unauthenticated());
  }
}
