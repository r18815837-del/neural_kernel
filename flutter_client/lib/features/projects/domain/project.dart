/// Domain model for a project in the list view.
class ProjectListItem {
  final String id;
  final String name;
  final String status;
  final String statusLabel;
  final DateTime? createdAt;
  final List<String> features;
  final bool artifactAvailable;

  const ProjectListItem({
    required this.id,
    required this.name,
    required this.status,
    required this.statusLabel,
    this.createdAt,
    this.features = const [],
    this.artifactAvailable = false,
  });

  bool get isTerminal =>
      const {'completed', 'failed', 'archived', 'cancelled'}.contains(status);
}
