/// Domain model for artifact metadata.
class ArtifactMetadata {
  final String projectId;
  final String projectName;
  final String artifactName;
  final int sizeBytes;
  final String format; // zip | folder
  final String downloadUrl;
  final DateTime? createdAt;
  final int? version;

  const ArtifactMetadata({
    required this.projectId,
    required this.projectName,
    required this.artifactName,
    required this.sizeBytes,
    required this.format,
    required this.downloadUrl,
    this.createdAt,
    this.version,
  });
}
