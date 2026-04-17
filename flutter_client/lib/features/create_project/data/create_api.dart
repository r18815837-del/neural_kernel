import 'package:dio/dio.dart';

import '../../../core/api/api_endpoints.dart';
import '../domain/create_request.dart';
import 'create_dto.dart';

/// API client for project creation.
class CreateApi {
  final Dio _dio;

  CreateApi(this._dio);

  /// POST /generate — returns project_id to navigate to status screen.
  Future<CreateProjectResponseDto> generate(CreateProjectRequest request) async {
    final resp = await _dio.post(
      ApiEndpoints.generate,
      data: request.toJson(),
    );
    return CreateProjectResponseDto.fromJson(resp.data as Map<String, dynamic>);
  }
}
