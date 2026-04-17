import 'package:flutter/material.dart';

import '../domain/tech_stack.dart';

/// Tech stack display (only non-null entries).
class TechStackSection extends StatelessWidget {
  final TechStack techStack;

  const TechStackSection({super.key, required this.techStack});

  @override
  Widget build(BuildContext context) {
    final entries = techStack.entries;
    if (entries.isEmpty) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Tech Stack', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        ...entries.entries.map(
          (e) => Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: Row(
              children: [
                SizedBox(
                  width: 100,
                  child: Text(
                    e.key,
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: Colors.grey.shade600),
                  ),
                ),
                Expanded(
                  child: Text(
                    e.value,
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
