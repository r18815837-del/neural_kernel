import 'package:dio/dio.dart';

import '../../../core/api/api_endpoints.dart';
import 'status_dto.dart';

/// API client for project status.
class StatusApi {
  final Dio _dio;

  StatusApi(this._dio);

  Future<ProjectStatusDto> getStatus(String projectId) async {
    final resp = await _dio.get(ApiEndpoints.clientStatus(projectId));
    return ProjectStatusDto.fromJson(resp.data as Map<String, dynamic>);
  }
}
