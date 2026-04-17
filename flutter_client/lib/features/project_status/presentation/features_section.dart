import 'package:flutter/material.dart';

import '../../../shared/widgets/nk_chip.dart';
import '../domain/feature_item.dart';

/// Feature chips list.
class FeaturesSection extends StatelessWidget {
  final List<FeatureItem> features;

  const FeaturesSection({super.key, required this.features});

  @override
  Widget build(BuildContext context) {
    if (features.isEmpty) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Features', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 6,
          children: features.map((f) {
            final color = switch (f.priority) {
              FeaturePriority.high => Colors.red,
              FeaturePriority.medium => Colors.blue,
              FeaturePriority.low => Colors.grey,
            };
            return Tooltip(
              message: f.description.isNotEmpty ? f.description : f.name,
              child: NKChip(label: f.name, color: color),
            );
          }).toList(),
        ),
      ],
    );
  }
}
