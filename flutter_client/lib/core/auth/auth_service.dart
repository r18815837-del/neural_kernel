import 'package:dio/dio.dart';

import '../api/api_endpoints.dart';
import 'auth_models.dart';
import 'auth_storage.dart';

/// Handles credential lifecycle: save, load, clear, validate.
class AuthService {
  final AuthStorage _storage;

  AuthService([AuthStorage? storage]) : _storage = storage ?? AuthStorage();

  Future<AuthCredentials?> loadCredentials() => _storage.load();
  Future<void> saveCredentials(AuthCredentials credentials) => _storage.save(credentials);
  Future<void> clearCredentials() => _storage.clear();
  Future<bool> hasCredentials() => _storage.hasCredentials();

  /// Validate credentials by hitting health + a protected endpoint.
  /// Returns `null` on success, or an error message string on failure.
  Future<String?> validate(AuthCredentials credentials) async {
    final dio = Dio(
      BaseOptions(
        baseUrl: credentials.baseUrl,
        connectTimeout: const Duration(seconds: 10),
        receiveTimeout: const Duration(seconds: 10),
      ),
    );

    // 1. Check server reachable.
    try {
      final healthResp = await dio.get(ApiEndpoints.health);
      if (healthResp.statusCode != 200) {
        return 'Server returned ${healthResp.statusCode}';
      }
    } on DioException catch (e) {
      if (e.type == DioExceptionType.connectionError ||
          e.type == DioExceptionType.connectionTimeout) {
        return 'Cannot connect to ${credentials.baseUrl}';
      }
      return 'Health check failed: ${e.message}';
    }

    // 2. Check auth works (skip for no-auth mode).
    if (credentials.mode != AuthMode.none) {
      try {
        final headers = <String, String>{};
        switch (credentials.mode) {
          case AuthMode.apiKey:
            headers['X-API-Key'] = credentials.secret;
          case AuthMode.bearer:
            headers['Authorization'] = 'Bearer ${credentials.secret}';
          case AuthMode.none:
            break;
        }
        await dio.get(
          ApiEndpoints.clientProjects,
          queryParameters: {'limit': 1},
          options: Options(headers: headers),
        );
      } on DioException catch (e) {
        if (e.response?.statusCode == 401) return 'Invalid credentials';
        if (e.response?.statusCode == 403) return 'Access denied';
        // Other errors — server is reachable and auth likely passed.
      }
    }

    return null; // success
  }
}
