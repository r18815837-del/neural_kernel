/// DTO matching backend ArtifactMetadataDTO.
class ArtifactMetadataDto {
  final String projectId;
  final String projectName;
  final String artifactName;
  final int artifactSizeBytes;
  final String packagingFormat;
  final String downloadUrl;
  final List<Map<String, dynamic>> features;
  final Map<String, dynamic>? techStack;
  final String createdAt;
  final int? version;

  ArtifactMetadataDto({
    required this.projectId,
    required this.projectName,
    required this.artifactName,
    required this.artifactSizeBytes,
    required this.packagingFormat,
    required this.downloadUrl,
    this.features = const [],
    this.techStack,
    this.createdAt = '',
    this.version,
  });

  factory ArtifactMetadataDto.fromJson(Map<String, dynamic> json) {
    return ArtifactMetadataDto(
      projectId: json['project_id'] as String? ?? '',
      projectName: json['project_name'] as String? ?? '',
      artifactName: json['artifact_name'] as String? ?? '',
      artifactSizeBytes: json['artifact_size_bytes'] as int? ?? 0,
      packagingFormat: json['packaging_format'] as String? ?? 'zip',
      downloadUrl: json['download_url'] as String? ?? '',
      features: (json['features'] as List<dynamic>?)
              ?.map((e) => e as Map<String, dynamic>)
              .toList() ??
          [],
      techStack: json['tech_stack'] as Map<String, dynamic>?,
      createdAt: json['created_at'] as String? ?? '',
      version: json['version'] as int?,
    );
  }
}
