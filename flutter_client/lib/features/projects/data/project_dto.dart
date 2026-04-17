/// DTOs matching the backend ProjectListResponseDTO / ProjectListItemDTO.
class ProjectListItemDto {
  final String projectId;
  final String projectName;
  final String status;
  final String statusLabel;
  final String createdAt;
  final List<String> features;
  final bool artifactAvailable;

  ProjectListItemDto({
    required this.projectId,
    required this.projectName,
    required this.status,
    required this.statusLabel,
    required this.createdAt,
    this.features = const [],
    this.artifactAvailable = false,
  });

  factory ProjectListItemDto.fromJson(Map<String, dynamic> json) {
    return ProjectListItemDto(
      projectId: json['project_id'] as String? ?? '',
      projectName: json['project_name'] as String? ?? '',
      status: json['status'] as String? ?? 'unknown',
      statusLabel: json['status_label'] as String? ?? '',
      createdAt: json['created_at'] as String? ?? '',
      features: (json['features'] as List<dynamic>?)
              ?.map((e) => e.toString())
              .toList() ??
          [],
      artifactAvailable: json['artifact_available'] as bool? ?? false,
    );
  }
}

class ProjectListResponseDto {
  final List<ProjectListItemDto> projects;
  final int total;
  final int limit;
  final int offset;

  ProjectListResponseDto({
    required this.projects,
    this.total = 0,
    this.limit = 50,
    this.offset = 0,
  });

  factory ProjectListResponseDto.fromJson(Map<String, dynamic> json) {
    return ProjectListResponseDto(
      projects: (json['projects'] as List<dynamic>?)
              ?.map((e) => ProjectListItemDto.fromJson(e as Map<String, dynamic>))
              .toList() ??
          [],
      total: json['total'] as int? ?? 0,
      limit: json['limit'] as int? ?? 50,
      offset: json['offset'] as int? ?? 0,
    );
  }
}
