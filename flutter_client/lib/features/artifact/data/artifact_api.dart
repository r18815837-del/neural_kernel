import 'package:dio/dio.dart';

import '../../../core/api/api_endpoints.dart';
import 'artifact_dto.dart';

/// API client for artifact metadata and binary download.
class ArtifactApi {
  final Dio _dio;

  ArtifactApi(this._dio);

  /// GET /client/download/{id}/info
  Future<ArtifactMetadataDto> getMetadata(String projectId) async {
    final resp = await _dio.get(ApiEndpoints.clientDownloadInfo(projectId));
    return ArtifactMetadataDto.fromJson(resp.data as Map<String, dynamic>);
  }

  /// GET /download/{id} — binary stream download with progress.
  Future<void> downloadArtifact({
    required String projectId,
    required String savePath,
    required void Function(int received, int total) onProgress,
    CancelToken? cancelToken,
  }) async {
    await _dio.download(
      ApiEndpoints.downloadArtifact(projectId),
      savePath,
      onReceiveProgress: onProgress,
      cancelToken: cancelToken,
    );
  }
}
