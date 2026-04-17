import 'package:dio/dio.dart';

import '../../../core/api/api_endpoints.dart';
import '../domain/ask_models.dart';

/// API client for the cognition /ask endpoint.
class AskApi {
  final Dio _dio;

  AskApi(this._dio);

  Future<AskResult> ask(String question) async {
    final response = await _dio.post(
      ApiEndpoints.ask,
      data: {'question': question},
    );
    return AskResult.fromJson(response.data as Map<String, dynamic>);
  }
}
