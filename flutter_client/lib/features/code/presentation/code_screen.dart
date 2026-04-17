import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../shared/widgets/nk_app_bar.dart';
import '../../../core/api/api_client.dart';
import '../data/code_api.dart';
import '../domain/code_models.dart';

class CodeScreen extends ConsumerStatefulWidget {
  const CodeScreen({super.key});

  @override
  ConsumerState<CodeScreen> createState() => _CodeScreenState();
}

class _CodeScreenState extends ConsumerState<CodeScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  final _codeController = TextEditingController();
  final _questionController = TextEditingController();

  bool _loading = false;

  // Results
  CodeAnalysisResult? _analysisResult;
  CodeRunResult? _runResult;
  CodeAskResult? _askResult;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    _codeController.dispose();
    _questionController.dispose();
    super.dispose();
  }

  CodeApi get _api => CodeApi(ref.read(apiClientProvider));

  Future<void> _analyze() async {
    if (_codeController.text.trim().isEmpty) return;
    setState(() => _loading = true);
    try {
      final result = await _api.analyze(_codeController.text);
      if (mounted) setState(() { _analysisResult = result; _loading = false; });
    } catch (e) {
      if (mounted) {
        setState(() => _loading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    }
  }

  Future<void> _run() async {
    if (_codeController.text.trim().isEmpty) return;
    setState(() => _loading = true);
    try {
      final result = await _api.run(_codeController.text);
      if (mounted) setState(() { _runResult = result; _loading = false; });
    } catch (e) {
      if (mounted) {
        setState(() => _loading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    }
  }

  Future<void> _ask() async {
    if (_questionController.text.trim().isEmpty) return;
    setState(() => _loading = true);
    try {
      final result = await _api.ask(
        _questionController.text,
        code: _codeController.text,
      );
      if (mounted) setState(() { _askResult = result; _loading = false; });
    } catch (e) {
      if (mounted) {
        setState(() => _loading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: const NKAppBar(title: 'Code Assistant', showBack: true),
      body: Column(
        children: [
          // Code input area
          Container(
            height: 200,
            margin: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: const Color(0xFF1E1E1E),
              borderRadius: BorderRadius.circular(12),
            ),
            child: TextField(
              controller: _codeController,
              maxLines: null,
              expands: true,
              style: const TextStyle(
                fontFamily: 'monospace',
                fontSize: 13,
                color: Color(0xFFD4D4D4),
                height: 1.5,
              ),
              decoration: const InputDecoration(
                hintText: 'Paste your code here...',
                hintStyle: TextStyle(color: Color(0xFF6A6A6A)),
                border: InputBorder.none,
                contentPadding: EdgeInsets.all(14),
              ),
            ),
          ),

          // Action buttons
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Row(
              children: [
                _ActionButton(
                  icon: Icons.search,
                  label: 'Analyze',
                  color: Colors.blue,
                  onPressed: _loading ? null : _analyze,
                ),
                const SizedBox(width: 8),
                _ActionButton(
                  icon: Icons.play_arrow,
                  label: 'Run',
                  color: Colors.green,
                  onPressed: _loading ? null : _run,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: TextField(
                    controller: _questionController,
                    decoration: InputDecoration(
                      hintText: 'Ask about code...',
                      isDense: true,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 14,
                        vertical: 10,
                      ),
                      suffixIcon: IconButton(
                        icon: const Icon(Icons.send, size: 18),
                        onPressed: _loading ? null : _ask,
                      ),
                    ),
                    textInputAction: TextInputAction.send,
                    onSubmitted: (_) => _ask(),
                  ),
                ),
              ],
            ),
          ),

          if (_loading)
            const Padding(
              padding: EdgeInsets.all(16),
              child: LinearProgressIndicator(),
            ),

          const SizedBox(height: 8),

          // Tabs for results
          TabBar(
            controller: _tabController,
            tabs: const [
              Tab(text: 'Analysis'),
              Tab(text: 'Output'),
              Tab(text: 'Answer'),
            ],
          ),

          // Results area
          Expanded(
            child: TabBarView(
              controller: _tabController,
              children: [
                _AnalysisTab(result: _analysisResult),
                _OutputTab(result: _runResult),
                _AnswerTab(result: _askResult),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ------------------------------------------------------------------
// Action button
// ------------------------------------------------------------------

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback? onPressed;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.color,
    this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return ElevatedButton.icon(
      onPressed: onPressed,
      icon: Icon(icon, size: 16),
      label: Text(label, style: const TextStyle(fontSize: 12)),
      style: ElevatedButton.styleFrom(
        backgroundColor: color,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        minimumSize: Size.zero,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      ),
    );
  }
}

// ------------------------------------------------------------------
// Analysis tab
// ------------------------------------------------------------------

class _AnalysisTab extends StatelessWidget {
  final CodeAnalysisResult? result;
  const _AnalysisTab({this.result});

  @override
  Widget build(BuildContext context) {
    if (result == null) {
      return const Center(
        child: Text('Paste code and tap Analyze', style: TextStyle(color: Colors.grey)),
      );
    }

    final theme = Theme.of(context);
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Language + syntax
        Row(
          children: [
            _chip(result!.language ?? 'unknown', Colors.purple.shade100),
            const SizedBox(width: 8),
            _chip(
              result!.syntaxValid ? 'Valid syntax' : 'Syntax error',
              result!.syntaxValid ? Colors.green.shade100 : Colors.red.shade100,
            ),
          ],
        ),
        const SizedBox(height: 12),

        // Functions
        if (result!.functions.isNotEmpty) ...[
          Text('Functions', style: theme.textTheme.titleSmall),
          const SizedBox(height: 4),
          for (final f in result!.functions)
            Padding(
              padding: const EdgeInsets.only(left: 8, bottom: 2),
              child: Text(
                '${f['name']}(${(f['args'] as List?)?.join(', ') ?? ''}) — line ${f['line']}',
                style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
              ),
            ),
          const SizedBox(height: 12),
        ],

        // Issues
        if (result!.issues.isNotEmpty) ...[
          Text('Issues', style: theme.textTheme.titleSmall?.copyWith(color: Colors.red)),
          const SizedBox(height: 4),
          for (final issue in result!.issues)
            _issueCard(issue, Colors.red.shade50, Icons.error_outline),
          const SizedBox(height: 12),
        ],

        // Suggestions
        if (result!.suggestions.isNotEmpty) ...[
          Text('Suggestions', style: theme.textTheme.titleSmall?.copyWith(color: Colors.orange)),
          const SizedBox(height: 4),
          for (final sug in result!.suggestions)
            _issueCard(sug, Colors.orange.shade50, Icons.lightbulb_outline),
          const SizedBox(height: 12),
        ],

        if (result!.issues.isEmpty && result!.suggestions.isEmpty)
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.green.shade50,
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Row(
              children: [
                Icon(Icons.check_circle, color: Colors.green),
                SizedBox(width: 8),
                Text('No issues found — code looks clean!'),
              ],
            ),
          ),

        // Explanation
        if (result!.explanation.isNotEmpty) ...[
          const SizedBox(height: 12),
          Text('Explanation', style: theme.textTheme.titleSmall),
          const SizedBox(height: 4),
          Text(result!.explanation, style: theme.textTheme.bodySmall),
        ],
      ],
    );
  }

  Widget _chip(String label, Color bg) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Text(label, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500)),
    );
  }

  Widget _issueCard(String text, Color bg, IconData icon) {
    return Container(
      margin: const EdgeInsets.only(bottom: 4),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 16, color: Colors.grey.shade700),
          const SizedBox(width: 8),
          Expanded(child: Text(text, style: const TextStyle(fontSize: 12))),
        ],
      ),
    );
  }
}

// ------------------------------------------------------------------
// Output tab
// ------------------------------------------------------------------

class _OutputTab extends StatelessWidget {
  final CodeRunResult? result;
  const _OutputTab({this.result});

  @override
  Widget build(BuildContext context) {
    if (result == null) {
      return const Center(
        child: Text('Paste code and tap Run', style: TextStyle(color: Colors.grey)),
      );
    }

    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF1E1E1E),
        borderRadius: BorderRadius.circular(12),
      ),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  result!.success ? Icons.check_circle : Icons.error,
                  size: 16,
                  color: result!.success ? Colors.green : Colors.red,
                ),
                const SizedBox(width: 8),
                Text(
                  result!.success ? 'Success' : 'Error (exit ${result!.exitCode})',
                  style: TextStyle(
                    color: result!.success ? Colors.green : Colors.red,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                if (result!.timedOut) ...[
                  const SizedBox(width: 8),
                  const Text(
                    'TIMEOUT',
                    style: TextStyle(color: Colors.orange, fontSize: 11),
                  ),
                ],
              ],
            ),
            const SizedBox(height: 12),
            if (result!.stdout.isNotEmpty) ...[
              const Text(
                'stdout:',
                style: TextStyle(color: Color(0xFF6A9955), fontSize: 11),
              ),
              const SizedBox(height: 4),
              Text(
                result!.stdout,
                style: const TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 13,
                  color: Color(0xFFD4D4D4),
                ),
              ),
            ],
            if (result!.stderr.isNotEmpty) ...[
              const SizedBox(height: 8),
              const Text(
                'stderr:',
                style: TextStyle(color: Color(0xFFF44747), fontSize: 11),
              ),
              const SizedBox(height: 4),
              Text(
                result!.stderr,
                style: const TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 13,
                  color: Color(0xFFF44747),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

// ------------------------------------------------------------------
// Answer tab
// ------------------------------------------------------------------

class _AnswerTab extends StatelessWidget {
  final CodeAskResult? result;
  const _AnswerTab({this.result});

  @override
  Widget build(BuildContext context) {
    if (result == null) {
      return const Center(
        child: Text('Type a question and press send', style: TextStyle(color: Colors.grey)),
      );
    }

    final theme = Theme.of(context);
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text(result!.answer, style: theme.textTheme.bodyMedium),
        const SizedBox(height: 12),
        if (result!.tips.isNotEmpty) ...[
          Text('Tips', style: theme.textTheme.titleSmall),
          const SizedBox(height: 4),
          for (final tip in result!.tips)
            Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.lightbulb_outline, size: 14, color: Colors.amber),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(tip, style: theme.textTheme.bodySmall),
                  ),
                ],
              ),
            ),
        ],
        if (result!.language != null) ...[
          const SizedBox(height: 8),
          Text(
            'Detected language: ${result!.language}',
            style: TextStyle(fontSize: 11, color: Colors.grey.shade500),
          ),
        ],
      ],
    );
  }
}
