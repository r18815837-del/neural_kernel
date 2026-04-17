import 'package:dio/dio.dart';

import '../auth/auth_models.dart';
import '../auth/auth_storage.dart';
import 'api_exceptions.dart';

/// Injects auth headers (API key or Bearer token) into every request.
class AuthInterceptor extends Interceptor {
  final AuthStorage _storage;

  AuthInterceptor(this._storage);

  @override
  void onRequest(
    RequestOptions options,
    RequestInterceptorHandler handler,
  ) async {
    try {
      final credentials = await _storage.load();
      if (credentials != null && credentials.mode != AuthMode.none) {
        switch (credentials.mode) {
          case AuthMode.apiKey:
            options.headers['X-API-Key'] = credentials.secret;
          case AuthMode.bearer:
            options.headers['Authorization'] = 'Bearer ${credentials.secret}';
          case AuthMode.none:
            break; // no headers needed
        }
      }
    } catch (_) {
      // Storage read failed — proceed without auth headers.
    }
    handler.next(options);
  }
}

/// Transforms Dio errors into typed [ApiException] subclasses.
class ErrorInterceptor extends Interceptor {
  @override
  void onError(DioException err, ErrorInterceptorHandler handler) {
    final statusCode = err.response?.statusCode;
    final data = err.response?.data;

    final message = _extractMessage(data) ?? err.message ?? 'Unknown error';

    if (statusCode != null) {
      switch (statusCode) {
        case 400:
          throw BadRequestException(message, data: _asMap(data));
        case 401:
          throw UnauthorizedException(message);
        case 403:
          throw ForbiddenException(message);
        case 404:
          throw NotFoundException(message);
        case 409:
          throw ConflictException(message);
        default:
          if (statusCode >= 500) throw ServerException(message);
      }
    }

    switch (err.type) {
      case DioExceptionType.connectionTimeout:
      case DioExceptionType.sendTimeout:
      case DioExceptionType.receiveTimeout:
        throw const TimeoutException();
      case DioExceptionType.connectionError:
        throw const NetworkException();
      default:
        throw ApiException(statusCode: statusCode, message: message);
    }
  }

  String? _extractMessage(dynamic data) {
    if (data is Map<String, dynamic>) {
      return data['message'] as String? ?? data['detail'] as String?;
    }
    return null;
  }

  Map<String, dynamic>? _asMap(dynamic data) {
    if (data is Map<String, dynamic>) return data;
    return null;
  }
}

/// Logs requests in debug mode.
class LoggingInterceptor extends Interceptor {
  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) {
    debugPrint('[NK] ${options.method} ${options.uri}');
    handler.next(options);
  }

  @override
  void onResponse(Response response, ResponseInterceptorHandler handler) {
    debugPrint('[NK] ${response.statusCode} ← ${response.requestOptions.uri}');
    handler.next(response);
  }

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) {
    debugPrint('[NK] ERROR ${err.response?.statusCode} ← ${err.requestOptions.uri}');
    handler.next(err);
  }
}

// For debugPrint without importing flutter/foundation in a non-widget file.
void debugPrint(String msg) {
  // ignore: avoid_print
  print(msg);
}
