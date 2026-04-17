/// A detected feature in the generated project.
class FeatureItem {
  final String name;
  final String description;
  final FeaturePriority priority;

  const FeatureItem({
    required this.name,
    this.description = '',
    this.priority = FeaturePriority.medium,
  });
}

enum FeaturePriority {
  high,
  medium,
  low;

  static FeaturePriority fromString(String s) => switch (s.toLowerCase()) {
        'high' => FeaturePriority.high,
        'low' => FeaturePriority.low,
        _ => FeaturePriority.medium,
      };
}
