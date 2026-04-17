import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../core/utils/extensions.dart';
import '../../../shared/constants/strings.dart';
import '../../../shared/widgets/nk_app_bar.dart';
import 'create_providers.dart';

class CreateProjectScreen extends ConsumerStatefulWidget {
  const CreateProjectScreen({super.key});

  @override
  ConsumerState<CreateProjectScreen> createState() =>
      _CreateProjectScreenState();
}

class _CreateProjectScreenState extends ConsumerState<CreateProjectScreen> {
  final _textController = TextEditingController();
  String _outputFormat = 'zip';

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  void _submit() {
    final text = _textController.text.trim();
    if (text.length < 10) {
      context.showSnackBar('Please describe your project (at least 10 chars)');
      return;
    }
    ref.read(createProjectProvider.notifier).submit(
          text,
          outputFormat: _outputFormat,
        );
  }

  @override
  Widget build(BuildContext context) {
    final createState = ref.watch(createProjectProvider);

    // Navigate on success.
    ref.listen(createProjectProvider, (prev, next) {
      if (next.status == CreateStatus.success && next.projectId != null) {
        context.go('/projects/${next.projectId}');
      }
      if (next.status == CreateStatus.error && next.error != null) {
        context.showSnackBar(next.error!, isError: true);
      }
    });

    final isSubmitting = createState.status == CreateStatus.submitting;

    return Scaffold(
      appBar: const NKAppBar(
        title: Strings.createProject,
        showBack: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Request text
            Expanded(
              child: TextField(
                controller: _textController,
                maxLines: null,
                expands: true,
                textAlignVertical: TextAlignVertical.top,
                decoration: const InputDecoration(
                  hintText: Strings.enterRequest,
                  border: OutlineInputBorder(),
                  alignLabelWithHint: true,
                ),
                enabled: !isSubmitting,
              ),
            ),
            const SizedBox(height: 12),

            // Output format selector
            Row(
              children: [
                Text(
                  'Output format:',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
                const SizedBox(width: 12),
                SegmentedButton<String>(
                  segments: const [
                    ButtonSegment(value: 'zip', label: Text('ZIP')),
                    ButtonSegment(value: 'folder', label: Text('Folder')),
                  ],
                  selected: {_outputFormat},
                  onSelectionChanged: (s) =>
                      setState(() => _outputFormat = s.first),
                ),
              ],
            ),
            const SizedBox(height: 16),

            // Generate button
            FilledButton.icon(
              onPressed: isSubmitting ? null : _submit,
              icon: isSubmitting
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : const Icon(Icons.auto_awesome),
              label: Text(
                isSubmitting ? Strings.generating : Strings.generate,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
