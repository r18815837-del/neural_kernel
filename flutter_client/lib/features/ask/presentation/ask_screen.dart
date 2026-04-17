import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../shared/widgets/nk_app_bar.dart';
import '../ask_providers.dart';
import '../domain/ask_models.dart';

class AskScreen extends ConsumerStatefulWidget {
  const AskScreen({super.key});

  @override
  ConsumerState<AskScreen> createState() => _AskScreenState();
}

class _AskScreenState extends ConsumerState<AskScreen> {
  final _controller = TextEditingController();
  final _scrollController = ScrollController();
  final _history = <AskResult>[];
  bool _loading = false;

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final text = _controller.text.trim();
    if (text.isEmpty || _loading) return;

    setState(() => _loading = true);

    try {
      final api = ref.read(askApiProvider);
      final result = await api.ask(text);

      if (!mounted) return;
      setState(() {
        _history.add(result);
        _loading = false;
        _controller.clear();
      });

      Future.delayed(const Duration(milliseconds: 100), () {
        if (_scrollController.hasClients) {
          _scrollController.animateTo(
            _scrollController.position.maxScrollExtent,
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOut,
          );
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _loading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: const NKAppBar(title: 'Ask NK', showBack: true),
      body: Column(
        children: [
          Expanded(
            child: _history.isEmpty
                ? Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.psychology_outlined,
                          size: 64,
                          color: theme.colorScheme.primary
                              .withValues(alpha: 0.4),
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'Neural Kernel Cognition',
                          style: theme.textTheme.titleMedium?.copyWith(
                            color: Colors.grey.shade600,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Ask anything — watch the pipeline think',
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: Colors.grey.shade400,
                          ),
                        ),
                      ],
                    ),
                  )
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.all(16),
                    itemCount: _history.length,
                    itemBuilder: (_, i) => _ChatBubble(
                      result: _history[i],
                    ),
                  ),
          ),

          // Input bar
          Container(
            padding: const EdgeInsets.fromLTRB(16, 8, 8, 16),
            decoration: BoxDecoration(
              color: theme.colorScheme.surface,
              border: Border(
                top: BorderSide(color: Colors.grey.shade200),
              ),
            ),
            child: SafeArea(
              top: false,
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _controller,
                      decoration: InputDecoration(
                        hintText: 'Type your question...',
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(24),
                        ),
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 20,
                          vertical: 12,
                        ),
                        isDense: true,
                      ),
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _send(),
                      enabled: !_loading,
                    ),
                  ),
                  const SizedBox(width: 8),
                  _loading
                      ? const Padding(
                          padding: EdgeInsets.all(12),
                          child: SizedBox(
                            width: 24,
                            height: 24,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          ),
                        )
                      : IconButton.filled(
                          onPressed: _send,
                          icon: const Icon(Icons.send_rounded),
                        ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ------------------------------------------------------------------
// Chat bubble with animated pipeline trace
// ------------------------------------------------------------------

class _ChatBubble extends StatefulWidget {
  final AskResult result;

  const _ChatBubble({required this.result});

  @override
  State<_ChatBubble> createState() => _ChatBubbleState();
}

class _ChatBubbleState extends State<_ChatBubble> {
  bool _traceExpanded = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final result = widget.result;

    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // User question
          Align(
            alignment: Alignment.centerRight,
            child: Container(
              constraints: BoxConstraints(
                maxWidth: MediaQuery.of(context).size.width * 0.75,
              ),
              padding: const EdgeInsets.symmetric(
                horizontal: 16,
                vertical: 10,
              ),
              decoration: BoxDecoration(
                color: theme.colorScheme.primary,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(16),
                  topRight: Radius.circular(16),
                  bottomLeft: Radius.circular(16),
                  bottomRight: Radius.circular(4),
                ),
              ),
              child: Text(
                result.question,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onPrimary,
                ),
              ),
            ),
          ),

          const SizedBox(height: 8),

          // NK answer
          Align(
            alignment: Alignment.centerLeft,
            child: Container(
              constraints: BoxConstraints(
                maxWidth: MediaQuery.of(context).size.width * 0.85,
              ),
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(4),
                  topRight: Radius.circular(16),
                  bottomLeft: Radius.circular(16),
                  bottomRight: Radius.circular(16),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Answer text
                  Text(
                    result.answer,
                    style: theme.textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 10),

                  // Tags: confidence + sources
                  Wrap(
                    spacing: 6,
                    runSpacing: 4,
                    children: [
                      _ConfidenceChip(confidence: result.confidence),
                      for (final s in result.sources)
                        _SourceChip(source: s),
                    ],
                  ),

                  // Pipeline mini-bar (always visible)
                  if (result.steps.isNotEmpty) ...[
                    const SizedBox(height: 10),
                    _PipelineMiniBar(steps: result.steps),
                  ],

                  // Expandable detailed trace
                  if (result.steps.isNotEmpty) ...[
                    const SizedBox(height: 6),
                    GestureDetector(
                      onTap: () => setState(
                        () => _traceExpanded = !_traceExpanded,
                      ),
                      child: Row(
                        children: [
                          Icon(
                            _traceExpanded
                                ? Icons.expand_less
                                : Icons.expand_more,
                            size: 16,
                            color: Colors.grey.shade500,
                          ),
                          const SizedBox(width: 4),
                          Text(
                            _traceExpanded
                                ? 'Hide trace'
                                : 'Show trace',
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: Colors.grey.shade500,
                              fontSize: 11,
                            ),
                          ),
                        ],
                      ),
                    ),
                    AnimatedCrossFade(
                      firstChild: const SizedBox.shrink(),
                      secondChild: _TracePanel(steps: result.steps),
                      crossFadeState: _traceExpanded
                          ? CrossFadeState.showSecond
                          : CrossFadeState.showFirst,
                      duration: const Duration(milliseconds: 250),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ------------------------------------------------------------------
// Confidence chip
// ------------------------------------------------------------------

class _ConfidenceChip extends StatelessWidget {
  final String confidence;
  const _ConfidenceChip({required this.confidence});

  @override
  Widget build(BuildContext context) {
    final (color, icon) = switch (confidence) {
      'high' => (Colors.green, Icons.verified_rounded),
      'medium' => (Colors.orange, Icons.check_circle_outline),
      'low' => (Colors.red.shade400, Icons.warning_amber_rounded),
      _ => (Colors.grey, Icons.help_outline),
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 12, color: color),
          const SizedBox(width: 4),
          Text(
            confidence,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}

// ------------------------------------------------------------------
// Source chip
// ------------------------------------------------------------------

class _SourceChip extends StatelessWidget {
  final String source;
  const _SourceChip({required this.source});

  @override
  Widget build(BuildContext context) {
    final (icon, label) = switch (source) {
      'wikipedia' => (Icons.language, 'Wikipedia'),
      'memory' => (Icons.memory, 'Memory'),
      'builtin_knowledge' => (Icons.auto_awesome, 'Built-in'),
      'neural_kernel_lm' => (Icons.smart_toy_outlined, 'NK LM'),
      'fallback' => (Icons.question_mark, 'Fallback'),
      _ when source.startsWith('specialist:') => (
          Icons.science,
          source.replaceFirst('specialist:', ''),
        ),
      _ => (Icons.source, source),
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: Colors.blue.shade50,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.blue.shade200),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 12, color: Colors.blue.shade700),
          const SizedBox(width: 4),
          Text(
            label,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w500,
              color: Colors.blue.shade700,
            ),
          ),
        ],
      ),
    );
  }
}

// ------------------------------------------------------------------
// Pipeline mini-bar — shows stages as colored dots
// ------------------------------------------------------------------

class _PipelineMiniBar extends StatelessWidget {
  final List<StepInfo> steps;
  const _PipelineMiniBar({required this.steps});

  static const _stageOrder = [
    'CLASSIFY',
    'RECALL',
    'SEARCH',
    'REASON',
    'VALIDATE',
    'PERSIST',
  ];

  @override
  Widget build(BuildContext context) {
    final completedKinds = steps.map((s) => s.kind).toSet();

    return Row(
      children: [
        for (var i = 0; i < _stageOrder.length; i++) ...[
          if (i > 0)
            Container(
              width: 12,
              height: 1.5,
              color: completedKinds.contains(_stageOrder[i])
                  ? Colors.green.shade300
                  : Colors.grey.shade300,
            ),
          _StageDot(
            label: _stageOrder[i],
            completed: completedKinds.contains(_stageOrder[i]),
          ),
        ],
      ],
    );
  }
}

class _StageDot extends StatelessWidget {
  final String label;
  final bool completed;
  const _StageDot({required this.label, required this.completed});

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: label,
      child: Container(
        width: 8,
        height: 8,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: completed ? Colors.green.shade400 : Colors.grey.shade300,
        ),
      ),
    );
  }
}

// ------------------------------------------------------------------
// Expandable trace panel — shows each stage in detail
// ------------------------------------------------------------------

class _TracePanel extends StatelessWidget {
  final List<StepInfo> steps;
  const _TracePanel({required this.steps});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Padding(
      padding: const EdgeInsets.only(top: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          for (var i = 0; i < steps.length; i++)
            _TraceRow(step: steps[i], isLast: i == steps.length - 1),
        ],
      ),
    );
  }
}

class _TraceRow extends StatelessWidget {
  final StepInfo step;
  final bool isLast;
  const _TraceRow({required this.step, required this.isLast});

  IconData _iconFor(String kind) => switch (kind) {
        'CLASSIFY' => Icons.category_outlined,
        'RECALL' => Icons.memory,
        'SEARCH' => Icons.search,
        'REASON' => Icons.lightbulb_outline,
        'VALIDATE' => Icons.verified_outlined,
        'PERSIST' => Icons.save_outlined,
        _ => Icons.circle_outlined,
      };

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final durationStr = step.durationMs != null
        ? '${step.durationMs!.toStringAsFixed(1)}ms'
        : '';

    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Timeline column
          SizedBox(
            width: 24,
            child: Column(
              children: [
                Icon(_iconFor(step.kind), size: 14, color: Colors.green.shade600),
                if (!isLast)
                  Expanded(
                    child: Container(
                      width: 1.5,
                      color: Colors.green.shade200,
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(width: 8),
          // Content
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(
                        step.kind,
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          color: Colors.grey.shade700,
                          letterSpacing: 0.5,
                        ),
                      ),
                      if (durationStr.isNotEmpty) ...[
                        const SizedBox(width: 6),
                        Text(
                          durationStr,
                          style: TextStyle(
                            fontSize: 10,
                            color: Colors.grey.shade400,
                          ),
                        ),
                      ],
                    ],
                  ),
                  if (step.output.isNotEmpty)
                    Text(
                      step.output,
                      style: TextStyle(
                        fontSize: 11,
                        color: Colors.grey.shade600,
                      ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
