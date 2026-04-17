import 'feature_item.dart';
import 'quality_score.dart';
import 'tech_stack.dart';

/// Rich domain model for the project status detail screen.
class ProjectStatus {
  final String id;
  final String name;
  final String status;
  final String statusLabel;
  final String message;
  final DateTime? createdAt;
  final DateTime? updatedAt;
  final DateTime? completedAt;
  final int progressPercent;
  final bool artifactAvailable;
  final String? artifactName;
  final int? artifactSizeBytes;
  final String? downloadUrl;
  final List<FeatureItem> features;
  final TechStack? techStack;
  final QualityScore? quality;
  final bool? executionReady;
  final bool llmUsed;
  final int agentCount;
  final int successfulAgentCount;
  final int failedAgentCount;
  final String? error;

  const ProjectStatus({
    required this.id,
    required this.name,
    required this.status,
    required this.statusLabel,
    required this.message,
    this.createdAt,
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

  bool get isTerminal =>
      const {'completed', 'failed', 'archived', 'cancelled'}.contains(status);

  Duration get pollInterval => switch (status) {
        'pending' => const Duration(seconds: 2),
        'in_progress' when progressPercent < 50 => const Duration(seconds: 3),
        'in_progress' => const Duration(seconds: 5),
        _ => Duration.zero,
      };
}
