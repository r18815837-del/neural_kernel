// DTOs matching backend ProjectStatusDTO, FeatureItemDTO, TechStackDTO, QualityScoreDTO.

class FeatureItemDto {
  final String name;
  final String description;
  final String priority;

  FeatureItemDto({
    required this.name,
    this.description = '',
    this.priority = 'medium',
  });

  factory FeatureItemDto.fromJson(Map<String, dynamic> json) {
    return FeatureItemDto(
      name: json['name'] as String? ?? '',
      description: json['description'] as String? ?? '',
      priority: json['priority'] as String? ?? 'medium',
    );
  }
}

class TechStackDto {
  final String? backend;
  final String? frontend;
  final String? database;
  final String? mobile;
  final String? deployment;

  TechStackDto({
    this.backend,
    this.frontend,
    this.database,
    this.mobile,
    this.deployment,
  });

  factory TechStackDto.fromJson(Map<String, dynamic> json) {
    return TechStackDto(
      backend: json['backend'] as String?,
      frontend: json['frontend'] as String?,
      database: json['database'] as String?,
      mobile: json['mobile'] as String?,
      deployment: json['deployment'] as String?,
    );
  }
}

class QualityScoreDto {
  final bool? scaffoldValid;
  final bool? executionReady;
  final bool? consistencyOk;
  final double? overallScore;

  QualityScoreDto({
    this.scaffoldValid,
    this.executionReady,
    this.consistencyOk,
    this.overallScore,
  });

  factory QualityScoreDto.fromJson(Map<String, dynamic> json) {
    return QualityScoreDto(
      scaffoldValid: json['scaffold_valid'] as bool?,
      executionReady: json['execution_ready'] as bool?,
      consistencyOk: json['consistency_ok'] as bool?,
      overallScore: (json['overall_score'] as num?)?.toDouble(),
    );
  }
}

class ProjectStatusDto {
  final String projectId;
  final String projectName;
  final String status;
  final String statusLabel;
  final String message;
  final String createdAt;
  final String? updatedAt;
  final String? completedAt;
  final int progressPercent;
  final bool artifactAvailable;
  final String? artifactName;
  final int? artifactSizeBytes;
  final String? downloadUrl;
  final List<FeatureItemDto> features;
  final TechStackDto? techStack;
  final QualityScoreDto? quality;
  final bool? executionReady;
  final bool llmUsed;
  final int agentCount;
  final int successfulAgentCount;
  final int failedAgentCount;
  final String? error;

  ProjectStatusDto({
    required this.projectId,
    required this.projectName,
    required this.status,
    required this.statusLabel,
    required this.message,
    required this.createdAt,
    this.updatedAt,
    this.completedAt,
    this.progressPercent = 0,
    this.artifactAvailable = false,
    this.artifactName,
    this.artifactSizeBytes,
    this.downloadUrl,
    this.features = const [],
    this.techStack,
    this.quality,
    this.executionReady,
    this.llmUsed = false,
    this.agentCount = 0,
    this.successfulAgentCount = 0,
    this.failedAgentCount = 0,
    this.error,
  });

  factory ProjectStatusDto.fromJson(Map<String, dynamic> json) {
    return ProjectStatusDto(
      projectId: json['project_id'] as String? ?? '',
      projectName: json['project_name'] as String? ?? '',
      status: json['status'] as String? ?? 'unknown',
      statusLabel: json['status_label'] as String? ?? '',
      message: json['message'] as String? ?? '',
      createdAt: json['created_at'] as String? ?? '',
      updatedAt: json['updated_at'] as String?,
      completedAt: json['completed_at'] as String?,
      progressPercent: json['progress_percent'] as int? ?? 0,
      artifactAvailable: json['artifact_available'] as bool? ?? false,
      artifactName: json['artifact_name'] as String?,
      artifactSizeBytes: json['artifact_size_bytes'] as int?,
      downloadUrl: json['download_url'] as String?,
      features: (json['features'] as List<dynamic>?)
              ?.map((e) => FeatureItemDto.fromJson(e as Map<String, dynamic>))
              .toList() ??
          [],
      techStack: json['tech_stack'] != null
          ? TechStackDto.fromJson(json['tech_stack'] as Map<String, dynamic>)
          : null,
      quality: json['quality'] != null
          ? QualityScoreDto.fromJson(json['quality'] as Map<String, dynamic>)
          : null,
      executionReady: json['execution_ready'] as bool?,
      llmUsed: json['llm_used'] as bool? ?? false,
      agentCount: json['agent_count'] as int? ?? 0,
      successfulAgentCount: json['successful_agent_count'] as int? ?? 0,
      failedAgentCount: json['failed_agent_count'] as int? ?? 0,
      error: json['error'] as String?,
    );
  }
}
