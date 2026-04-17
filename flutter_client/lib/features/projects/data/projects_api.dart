import 'package:dio/dio.dart';

import '../../../core/api/api_endpoints.dart';
import 'project_dto.dart';

/// API client for the project list.
class ProjectsApi {
  final Dio _dio;

  ProjectsApi(this._dio);

  Future<ProjectListResponseDto> getProjects({
    int limit = 50,
    int offset = 0,
    String? userId,
  }) async {
    final params = <String, dynamic>{
      'limit': limit,
      'offset': offset,
    };
    if (userId != null) params['user_id'] = userId;

    final resp = await _dio.get(
      ApiEndpoints.clientProjects,
      queryParameters: params,
    );
    return ProjectListResponseDto.fromJson(resp.data as Map<String, dynamic>);
  }
}
