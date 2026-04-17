import 'package:dio/dio.dart';

import '../../../core/api/api_endpoints.dart';
import '../domain/code_models.dart';

/// API client for the code assistant endpoints.
class CodeApi {
  final Dio _dio;

  CodeApi(this._dio);

  Future<CodeAnalysisResult> analyze(String code, {String question = ''}) async {
    final response = await _dio.post(
      ApiEndpoints.codeAnalyze,
      data: {'code': code, 'question': question},
    );
    return CodeAnalysisResult.fromJson(response.data as Map<String, dynamic>);
  }

  Future<CodeRunResult> run(String code) async {
    final response = await _dio.post(
      ApiEndpoints.codeRun,
      data: {'code': code},
    );
    return CodeRunResult.fromJson(response.data as Map<String, dynamic>);
  }

  Future<CodeAskResult> ask(String question, {String code = ''}) async {
    final response = await _dio.post(
      ApiEndpoints.codeAsk,
      data: {'question': question, 'code': code},
    );
    return CodeAskResult.fromJson(response.data as Map<String, dynamic>);
  }
}
