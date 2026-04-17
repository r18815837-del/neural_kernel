import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/api/api_client.dart';
import '../../core/utils/date_utils.dart';
import 'data/artifact_api.dart';
import 'data/download_service.dart';
import 'domain/artifact_metadata.dart';

/// Artifact API instance.
final artifactApiProvider = Provider<ArtifactApi>((ref) {
  return ArtifactApi(ref.watch(apiClientProvider));
});

/// Download service instance.
final downloadServiceProvider = Provider<DownloadService>((ref) {
  return DownloadService(ref.watch(artifactApiProvider));
});

/// Artifact metadata — fetched once per project.
final artifactMetaProvider =
    FutureProvider.autoDispose.family<ArtifactMetadata, String>(
  (ref, projectId) async {
    final api = ref.watch(artifactApiProvider);
    final dto = await api.getMetadata(projectId);
    return ArtifactMetadata(
      projectId: dto.projectId,
      projectName: dto.projectName,
      artifactName: dto.artifactName,
      sizeBytes: dto.artifactSizeBytes,
      format: dto.packagingFormat,
      downloadUrl: dto.downloadUrl,
      createdAt: NKDateUtils.tryParse(dto.createdAt),
      version: dto.version,
    );
  },
);

/// Download state stream.
final downloadStateProvider =
    StreamProvider.autoDispose.family<DownloadState, String>(
  (ref, projectId) {
    final service = ref.watch(downloadServiceProvider);
    ref.onDispose(() => service.cancel());
    return service.download(projectId);
  },
);
