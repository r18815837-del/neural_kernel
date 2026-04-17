import 'package:dio/dio.dart';

import '../../../core/api/api_endpoints.dart';

/// API calls for auth validation.
class AuthApi {
  final Dio _dio;

  AuthApi(this._dio);

  /// Check if the server is reachable.
  Future<bool> healthCheck() async {
    final resp = await _dio.get(ApiEndpoints.health);
    return resp.statusCode == 200;
  }

  /// Validate credentials by fetching the project list.
  Future<bool> validateAuth() async {
    try {
      await _dio.get(
        ApiEndpoints.clientProjects,
        queryParameters: {'limit': 1},
      );
      return true;
    } on DioException catch (e) {
      if (e.response?.statusCode == 401 || e.response?.statusCode == 403) {
        return false;
      }
      // Other errors (e.g. no projects) are still valid auth.
      return true;
    }
  }
}
