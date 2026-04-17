import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../auth/auth_storage.dart';
import 'api_interceptors.dart';

/// Current base URL — updated when user connects.
final baseUrlProvider = StateProvider<String>((ref) => '');

/// Provides a configured [Dio] instance with auth + error interceptors.
final apiClientProvider = Provider<Dio>((ref) {
  final baseUrl = ref.watch(baseUrlProvider);
  final storage = AuthStorage();

  final dio = Dio(
    BaseOptions(
      baseUrl: baseUrl,
      connectTimeout: const Duration(seconds: 15),
      receiveTimeout: const Duration(seconds: 30),
      sendTimeout: const Duration(seconds: 15),
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    ),
  );

  dio.interceptors.addAll([
    AuthInterceptor(storage),
    ErrorInterceptor(),
    LoggingInterceptor(),
  ]);

  return dio;
});
