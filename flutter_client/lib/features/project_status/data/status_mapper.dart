import '../../../core/utils/date_utils.dart';
import '../domain/feature_item.dart';
import '../domain/project_status.dart';
import '../domain/quality_score.dart';
import '../domain/tech_stack.dart';
import 'status_dto.dart';

/// Maps backend ProjectStatusDto to domain ProjectStatus.
class StatusMapper {
  StatusMapper._();

  static ProjectStatus toDomain(ProjectStatusDto dto) {
    return ProjectStatus(
      id: dto.projectId,
      name: dto.projectName,
      status: dto.status,
      statusLabel: dto.statusLabel,
      message: dto.message,
      createdAt: NKDateUtils.tryParse(dto.createdAt),
      updatedAt: NKDateUtils.tryParse(dto.updatedAt),
      completedAt: NKDateUtils.tryParse(dto.completedAt),
      progressPercent: dto.progressPercent,
      artifactAvailable: dto.artifactAvailable,
      artifactName: dto.artifactName,
      artifactSizeBytes: dto.artifactSizeBytes,
      downloadUrl: dto.downloadUrl,
      features: dto.features.map(_mapFeature).toList(),
      techStack: dto.techStack != null ? _mapTechStack(dto.techStack!) : null,
      quality: dto.quality != null ? _mapQuality(dto.quality!) : null,
      executionReady: dto.executionReady,
      llmUsed: dto.llmUsed,
      agentCount: dto.agentCount,
      successfulAgentCount: dto.successfulAgentCount,
      failedAgentCount: dto.failedAgentCount,
      error: dto.error,
    );
  }

  static FeatureItem _mapFeature(FeatureItemDto dto) {
    return FeatureItem(
      name: dto.name,
      description: dto.description,
      priority: FeaturePriority.fromString(dto.priority),
    );
  }

  static TechStack _mapTechStack(TechStackDto dto) {
    return TechStack(
      backend: dto.backend,
      frontend: dto.frontend,
      database: dto.database,
      mobile: dto.mobile,
      deployment: dto.deployment,
    );
  }

  static QualityScore _mapQuality(QualityScoreDto dto) {
    return QualityScore(
      scaffoldValid: dto.scaffoldValid,
      executionReady: dto.executionReady,
      consistencyOk: dto.consistencyOk,
      overallScore: dto.overallScore,
    );
  }
}
