import 'dart:io';

import 'package:desktop_drop/desktop_drop.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'chat_controller.dart';
import 'main.dart';
import 'models.dart';
import 'widgets.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  // Видимость панелей — chat, code, sidebar.
  bool _showSidebar = true;
  bool _showCode = true;

  // Фракция ширины, которую занимает код-панель (от Row-пространства
  // после sidebar). Пользователь может тянуть разделитель.
  double _codeFraction = 0.4;

  @override
  Widget build(BuildContext context) {
    final codeFlex = (_codeFraction * 1000).round().clamp(150, 850);
    final chatFlex = 1000 - codeFlex;

    return Scaffold(
      body: Stack(
        children: [
          Row(
            children: [
              if (_showSidebar) ...[
                const SidebarPanel(width: 260),
                const VerticalDivider(width: 1),
              ],
              Expanded(
                flex: _showCode ? chatFlex : 1000,
                child: _ChatArea(
                  toggles: _PanelToggles(
                    showSidebar: _showSidebar,
                    showCode: _showCode,
                    onSidebar: () =>
                        setState(() => _showSidebar = !_showSidebar),
                    onCode: () => setState(() => _showCode = !_showCode),
                  ),
                ),
              ),
              if (_showCode) ...[
                _SplitHandle(
                  onDrag: (delta) {
                    final width = MediaQuery.of(context).size.width;
                    final available = width - (_showSidebar ? 260 : 0);
                    setState(() {
                      _codeFraction =
                          (_codeFraction - delta / available).clamp(0.15, 0.85);
                    });
                  },
                ),
                Expanded(
                  flex: codeFlex,
                  child: const CodePanel(),
                ),
              ],
            ],
          ),
          // Поверх — confirmation overlay для agent-tool calls.
          const ConfirmationOverlay(),
        ],
      ),
    );
  }
}

/// Параметры для панели togglов — передаём в _ChatArea, чтобы он
/// разместил их в шапке (рядом с model selector).
class _PanelToggles {
  _PanelToggles({
    required this.showSidebar,
    required this.showCode,
    required this.onSidebar,
    required this.onCode,
  });
  final bool showSidebar;
  final bool showCode;
  final VoidCallback onSidebar;
  final VoidCallback onCode;
}

/// Вертикальный разделитель с drag-hover эффектом — можно тянуть, чтобы
/// изменить соотношение chat/code.
class _SplitHandle extends StatefulWidget {
  const _SplitHandle({required this.onDrag});
  final void Function(double delta) onDrag;

  @override
  State<_SplitHandle> createState() => _SplitHandleState();
}

class _SplitHandleState extends State<_SplitHandle> {
  bool _hovered = false;
  bool _dragging = false;

  @override
  Widget build(BuildContext context) {
    final active = _hovered || _dragging;
    return MouseRegion(
      cursor: SystemMouseCursors.resizeColumn,
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        behavior: HitTestBehavior.translucent,
        onHorizontalDragStart: (_) => setState(() => _dragging = true),
        onHorizontalDragEnd: (_) => setState(() => _dragging = false),
        onHorizontalDragUpdate: (d) => widget.onDrag(d.delta.dx),
        child: Container(
          width: 6,
          color: active ? const Color(0xFF7C9CFF) : Colors.white12,
        ),
      ),
    );
  }
}

class _ChatArea extends StatefulWidget {
  const _ChatArea({this.toggles});
  final _PanelToggles? toggles;

  @override
  State<_ChatArea> createState() => _ChatAreaState();
}

class _ChatAreaState extends State<_ChatArea> {
  final _inputController = TextEditingController();
  final _scrollController = ScrollController();
  final _focusNode = FocusNode();

  // In-chat search state
  bool _searchOpen = false;
  final _inChatSearchCtrl = TextEditingController();
  final _inChatSearchFocus = FocusNode();
  void Function()? _openListener;
  void Function()? _closeListener;

  // File drop overlay
  bool _droppingFiles = false;

  @override
  void initState() {
    super.initState();
    InputFocus.node = _focusNode;
    _inputController.addListener(() => setState(() {}));
    _inChatSearchCtrl.addListener(() => setState(() {}));

    _openListener = () {
      setState(() => _searchOpen = true);
      WidgetsBinding.instance
          .addPostFrameCallback((_) => _inChatSearchFocus.requestFocus());
    };
    _closeListener = () {
      if (_searchOpen) {
        setState(() => _searchOpen = false);
      }
    };
    InChatSearchBus.onOpen(_openListener!);
    InChatSearchBus.onClose(_closeListener!);
  }

  @override
  void dispose() {
    if (_openListener != null) InChatSearchBus.offOpen(_openListener!);
    if (_closeListener != null) InChatSearchBus.offClose(_closeListener!);
    _inputController.dispose();
    _scrollController.dispose();
    _focusNode.dispose();
    _inChatSearchCtrl.dispose();
    _inChatSearchFocus.dispose();
    if (identical(InputFocus.node, _focusNode)) InputFocus.node = null;
    super.dispose();
  }

  static const double _stickyThreshold = 120;

  void _scrollToBottom({bool animated = true}) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      final pos = _scrollController.position;
      final maxScroll = pos.maxScrollExtent;
      final current = pos.pixels;
      if (maxScroll - current > _stickyThreshold) return;
      if (animated) {
        _scrollController.animateTo(
          maxScroll,
          duration: const Duration(milliseconds: 150),
          curve: Curves.easeOut,
        );
      } else {
        _scrollController.jumpTo(maxScroll);
      }
    });
  }

  Future<void> _handleSend() async {
    final text = _inputController.text;
    if (text.trim().isEmpty) return;
    _inputController.clear();
    final ctrl = context.read<ChatController>();
    await ctrl.sendUserMessage(text);
    _scrollToBottom();
    _focusNode.requestFocus();
  }

  Future<void> _handleFilesDropped(DropDoneDetails detail) async {
    setState(() => _droppingFiles = false);
    if (detail.files.isEmpty) return;

    final allowed = {'.py', '.md', '.txt', '.json', '.yaml', '.yml', '.dart',
        '.js', '.ts', '.tsx', '.jsx', '.rs', '.go', '.java', '.kt', '.cpp',
        '.c', '.h', '.hpp', '.sh', '.ps1', '.toml', '.cfg', '.ini', '.html',
        '.css', '.xml', '.sql', '.log'};

    final buffer = StringBuffer();
    if (_inputController.text.trim().isNotEmpty) {
      buffer.write(_inputController.text);
      buffer.writeln('\n');
    }

    for (final f in detail.files) {
      final path = f.path;
      final ext = _extension(path);
      if (!allowed.contains(ext.toLowerCase())) {
        buffer.writeln('_(skipped: ${_basename(path)} — unsupported type)_');
        continue;
      }
      try {
        final file = File(path);
        final size = await file.length();
        if (size > 256 * 1024) {
          buffer.writeln(
            '_(skipped: ${_basename(path)} — too large: '
            '${(size / 1024).toStringAsFixed(0)} KB)_',
          );
          continue;
        }
        final content = await file.readAsString();
        final lang = _langFromExt(ext);
        buffer.writeln('File `${_basename(path)}`:');
        buffer.writeln('```$lang');
        buffer.writeln(content.trim());
        buffer.writeln('```');
        buffer.writeln();
      } catch (e) {
        buffer.writeln('_(failed to read ${_basename(path)}: $e)_');
      }
    }

    _inputController.text = buffer.toString();
    _inputController.selection = TextSelection.fromPosition(
      TextPosition(offset: _inputController.text.length),
    );
    _focusNode.requestFocus();
  }

  String _extension(String path) {
    final i = path.lastIndexOf('.');
    if (i < 0) return '';
    return path.substring(i);
  }

  String _basename(String path) {
    final sep = path.contains('\\') ? '\\' : '/';
    return path.split(sep).last;
  }

  String _langFromExt(String ext) {
    return switch (ext.toLowerCase()) {
      '.py' => 'python',
      '.md' => 'markdown',
      '.json' => 'json',
      '.yaml' || '.yml' => 'yaml',
      '.dart' => 'dart',
      '.js' => 'javascript',
      '.ts' => 'typescript',
      '.jsx' => 'jsx',
      '.tsx' => 'tsx',
      '.rs' => 'rust',
      '.go' => 'go',
      '.java' => 'java',
      '.kt' => 'kotlin',
      '.cpp' || '.c' || '.h' || '.hpp' => 'cpp',
      '.sh' => 'bash',
      '.ps1' => 'powershell',
      '.html' => 'html',
      '.css' => 'css',
      '.xml' => 'xml',
      '.sql' => 'sql',
      '.toml' => 'toml',
      _ => '',
    };
  }

  @override
  Widget build(BuildContext context) {
    final ctrl = context.watch<ChatController>();
    final conv = ctrl.activeConversation;

    if (ctrl.lastError != null) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;
        final err = ctrl.lastError;
        if (err == null) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(err, maxLines: 3),
            backgroundColor: Colors.red.shade900,
            behavior: SnackBarBehavior.floating,
            duration: const Duration(seconds: 5),
          ),
        );
        ctrl.clearError();
      });
    }

    if (ctrl.streaming) _scrollToBottom(animated: false);

    return DropTarget(
      onDragEntered: (_) => setState(() => _droppingFiles = true),
      onDragExited: (_) => setState(() => _droppingFiles = false),
      onDragDone: _handleFilesDropped,
      child: Stack(
        children: [
          Column(
            children: [
              _Header(
                onExport: () => _exportChat(context),
                toggles: widget.toggles,
              ),
              if (_searchOpen)
                _InChatSearchBar(
                  controller: _inChatSearchCtrl,
                  focusNode: _inChatSearchFocus,
                  onClose: () {
                    setState(() => _searchOpen = false);
                    _inChatSearchCtrl.clear();
                  },
                  matchCount: _countMatches(conv, _inChatSearchCtrl.text),
                ),
              const Divider(height: 1),
              Expanded(
                child: conv == null
                    ? _EmptyState(
                        onSuggestionTap: (suggestion) {
                          _inputController.text = suggestion;
                          _inputController.selection =
                              TextSelection.fromPosition(
                            TextPosition(offset: suggestion.length),
                          );
                          _focusNode.requestFocus();
                        },
                      )
                    : _MessageList(
                        scrollController: _scrollController,
                        conversation: conv,
                        highlightQuery: _searchOpen
                            ? _inChatSearchCtrl.text
                            : '',
                      ),
              ),
              _InputBar(
                controller: _inputController,
                focusNode: _focusNode,
                onSend: _handleSend,
              ),
            ],
          ),
          // Drop overlay
          if (_droppingFiles)
            Positioned.fill(
              child: IgnorePointer(
                child: Container(
                  color: const Color(0xCC1E1F22),
                  alignment: Alignment.center,
                  child: Container(
                    padding: const EdgeInsets.all(32),
                    decoration: BoxDecoration(
                      color: const Color(0xFF2A2B2E),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: const Color(0xFF7C9CFF),
                        width: 2,
                      ),
                    ),
                    child: const Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.file_download_outlined,
                          size: 48,
                          color: Color(0xFF7C9CFF),
                        ),
                        SizedBox(height: 12),
                        Text(
                          'Отпусти файл сюда',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        SizedBox(height: 4),
                        Text(
                          'Контент попадёт в поле ввода',
                          style: TextStyle(
                            color: Colors.white54,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  int _countMatches(Conversation? conv, String query) {
    if (conv == null || query.trim().isEmpty) return 0;
    final q = query.toLowerCase();
    int count = 0;
    for (final m in conv.messages) {
      count += _countSubstring(m.content.toLowerCase(), q);
    }
    return count;
  }

  int _countSubstring(String text, String pattern) {
    if (pattern.isEmpty) return 0;
    int count = 0, start = 0;
    while (true) {
      final i = text.indexOf(pattern, start);
      if (i < 0) break;
      count++;
      start = i + pattern.length;
    }
    return count;
  }

  Future<void> _exportChat(BuildContext context) async {
    final ctrl = context.read<ChatController>();
    final md = ctrl.exportActiveAsMarkdown();
    if (md.isEmpty) return;
    await Clipboard.setData(ClipboardData(text: md));
    if (!context.mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Chat copied as markdown → clipboard'),
        duration: Duration(seconds: 2),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// In-chat search bar
// ════════════════════════════════════════════════════════════════════════
class _InChatSearchBar extends StatelessWidget {
  const _InChatSearchBar({
    required this.controller,
    required this.focusNode,
    required this.onClose,
    required this.matchCount,
  });

  final TextEditingController controller;
  final FocusNode focusNode;
  final VoidCallback onClose;
  final int matchCount;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      color: const Color(0xE81E1F22),
      child: Row(
        children: [
          const Icon(Icons.search, size: 16, color: Colors.white54),
          const SizedBox(width: 8),
          Expanded(
            child: TextField(
              controller: controller,
              focusNode: focusNode,
              style: const TextStyle(color: Colors.white, fontSize: 13),
              decoration: const InputDecoration(
                hintText: 'Find in chat…  (Esc to close)',
                border: InputBorder.none,
                filled: false,
                isDense: true,
                contentPadding: EdgeInsets.zero,
              ),
            ),
          ),
          const SizedBox(width: 8),
          if (controller.text.isNotEmpty)
            Text(
              '$matchCount match${matchCount == 1 ? '' : 'es'}',
              style: const TextStyle(color: Colors.white54, fontSize: 11),
            ),
          const SizedBox(width: 8),
          IconButton(
            icon: const Icon(Icons.close, size: 16),
            color: Colors.white54,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
            onPressed: onClose,
          ),
        ],
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// Header
// ════════════════════════════════════════════════════════════════════════
class _Header extends StatelessWidget {
  const _Header({required this.onExport, this.toggles});
  final VoidCallback onExport;
  final _PanelToggles? toggles;

  @override
  Widget build(BuildContext context) {
    final ctrl = context.watch<ChatController>();
    final conv = ctrl.activeConversation;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      color: const Color(0xE81E1F22),
      child: Row(
        children: [
          // Panel toggles — скрыть/показать sidebar и code panel.
          if (toggles != null) ...[
            PanelToggle(
              icon: Icons.view_sidebar_outlined,
              active: toggles!.showSidebar,
              tooltip: 'Toggle sidebar',
              onPressed: toggles!.onSidebar,
            ),
            PanelToggle(
              icon: Icons.code,
              active: toggles!.showCode,
              tooltip: 'Toggle code panel',
              onPressed: toggles!.onCode,
            ),
            const SizedBox(width: 8),
            const SizedBox(
              height: 20,
              child: VerticalDivider(width: 1, color: Colors.white12),
            ),
            const SizedBox(width: 8),
          ],
          if (conv != null) ...[
            Flexible(
              child: Text(
                conv.title,
                style: Theme.of(context).textTheme.titleMedium,
                overflow: TextOverflow.ellipsis,
              ),
            ),
            const SizedBox(width: 12),
          ],
          // Inline RAG progress
          if (ctrl.ragSearching) ...[
            const SizedBox(
              width: 14,
              height: 14,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: Color(0xFF7C9CFF),
              ),
            ),
            const SizedBox(width: 6),
            const Text(
              'Searching codebase…',
              style: TextStyle(color: Color(0xFF7C9CFF), fontSize: 11),
            ),
          ],
          const Spacer(),
          if (ctrl.streaming)
            TextButton.icon(
              onPressed: () => ctrl.cancelGeneration(),
              icon: const Icon(Icons.stop_circle_outlined, size: 18),
              label: const Text('Stop'),
              style: TextButton.styleFrom(foregroundColor: Colors.redAccent),
            ),
          const SizedBox(width: 8),
          const AgentToggle(),
          const SizedBox(width: 8),
          const RagToggle(),
          const SizedBox(width: 8),
          const ModelSelector(),
          const SizedBox(width: 4),
          IconButton(
            tooltip: 'Find in chat  ·  Ctrl+F',
            icon: const Icon(Icons.manage_search_outlined, size: 18),
            onPressed: () => InChatSearchBus.open(),
          ),
          IconButton(
            tooltip: 'Export as markdown  ·  Ctrl+Shift+E',
            icon: const Icon(Icons.ios_share_outlined, size: 18),
            onPressed: onExport,
          ),
          IconButton(
            tooltip: 'Settings',
            icon: const Icon(Icons.tune, size: 18),
            onPressed: () => _openSettings(context),
          ),
        ],
      ),
    );
  }

  void _openSettings(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: const Color(0xFF26272A),
      isScrollControlled: true,
      builder: (_) => const SettingsSheet(),
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// Empty state
// ════════════════════════════════════════════════════════════════════════
class _EmptyState extends StatelessWidget {
  const _EmptyState({required this.onSuggestionTap});
  final void Function(String text) onSuggestionTap;

  static const List<({String icon, String title, String prompt})>
      _suggestions = [
    (
      icon: '🐍',
      title: 'Write a function',
      prompt:
          'Напиши Python-функцию, которая делает BPE-tokenization для строки. '
              'С type hints и docstring.',
    ),
    (
      icon: '🔍',
      title: 'Explain this code',
      prompt:
          'Объясни, как работает causal mask в scaled_dot_product_attention. '
              'Когда используется и что меняется без неё.',
    ),
    (
      icon: '🧪',
      title: 'Write a test',
      prompt:
          'Напиши pytest-тест для функции softmax в стиле tests_parity — '
              'с сравнением против torch.softmax.',
    ),
    (
      icon: '🪲',
      title: 'Find the bug',
      prompt: 'Посмотри на этот код и найди баг:\n\n```python\n'
          'def mean(xs):\n    return sum(xs) / len(xs)\n\n'
          'print(mean([]))\n```',
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 680),
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 56,
                height: 56,
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [Color(0xFF7C9CFF), Color(0xFF6FE0C5)],
                  ),
                  borderRadius: BorderRadius.circular(14),
                  boxShadow: [
                    BoxShadow(
                      color: const Color(0xFF7C9CFF).withOpacity(0.3),
                      blurRadius: 20,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: const Icon(Icons.auto_awesome,
                    color: Colors.white, size: 28),
              ),
              const SizedBox(height: 20),
              const Text(
                'NK Chat',
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.w600,
                  color: Colors.white,
                  letterSpacing: -0.3,
                ),
              ),
              const SizedBox(height: 6),
              const Text(
                'Локальный ассистент на твоём железе',
                style: TextStyle(color: Colors.white54, fontSize: 14),
              ),
              const SizedBox(height: 36),
              GridView.count(
                shrinkWrap: true,
                crossAxisCount: 2,
                mainAxisSpacing: 12,
                crossAxisSpacing: 12,
                childAspectRatio: 2.8,
                children: _suggestions
                    .map((s) => _SuggestionCard(
                          icon: s.icon,
                          title: s.title,
                          prompt: s.prompt,
                          onTap: () => onSuggestionTap(s.prompt),
                        ))
                    .toList(),
              ),
              const SizedBox(height: 16),
              const Text(
                'или перетащи файлы прямо сюда · Ctrl+L фокус, Ctrl+F поиск',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white38,
                  fontSize: 12,
                  fontStyle: FontStyle.italic,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _SuggestionCard extends StatefulWidget {
  const _SuggestionCard({
    required this.icon,
    required this.title,
    required this.prompt,
    required this.onTap,
  });

  final String icon;
  final String title;
  final String prompt;
  final VoidCallback onTap;

  @override
  State<_SuggestionCard> createState() => _SuggestionCardState();
}

class _SuggestionCardState extends State<_SuggestionCard> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      cursor: SystemMouseCursors.click,
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 120),
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color:
                _hovered ? const Color(0xFF2A2B2E) : const Color(0xFF202123),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: _hovered ? Colors.white24 : Colors.white10,
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Row(
                children: [
                  Text(widget.icon, style: const TextStyle(fontSize: 18)),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      widget.title,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 6),
              Text(
                widget.prompt,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  color: Colors.white54,
                  fontSize: 11,
                  height: 1.4,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _MessageList extends StatelessWidget {
  const _MessageList({
    required this.scrollController,
    required this.conversation,
    required this.highlightQuery,
  });

  final ScrollController scrollController;
  final Conversation conversation;
  final String highlightQuery;

  @override
  Widget build(BuildContext context) {
    final messages = conversation.messages;
    if (messages.isEmpty) {
      return const Center(
        child: Text(
          'Задай что-нибудь. Модель отвечает локально.',
          style: TextStyle(color: Colors.white38),
        ),
      );
    }

    return ListView.builder(
      controller: scrollController,
      padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 24),
      itemCount: messages.length,
      itemBuilder: (context, i) => RepaintBoundary(
        key: ValueKey(messages[i].id),
        child: MessageBubble(
          message: messages[i],
          highlightQuery: highlightQuery,
        ),
      ),
    );
  }
}

class _InputBar extends StatelessWidget {
  const _InputBar({
    required this.controller,
    required this.focusNode,
    required this.onSend,
  });

  final TextEditingController controller;
  final FocusNode focusNode;
  final VoidCallback onSend;

  int _estimateTokens(String text) {
    if (text.isEmpty) return 0;
    int count = 0;
    for (final ch in text.runes) {
      if (ch < 128) {
        count += 1;
      } else {
        count += 2;
      }
    }
    return (count / 4).ceil();
  }

  @override
  Widget build(BuildContext context) {
    final ctrl = context.watch<ChatController>();
    final conv = ctrl.activeConversation;

    int contextTokens = 0;
    if (conv != null) {
      contextTokens += _estimateTokens(conv.systemPrompt);
      for (final m in conv.messages) {
        contextTokens += _estimateTokens(m.content);
      }
    }
    contextTokens += _estimateTokens(controller.text);
    final numCtx = ctrl.contextWindow;

    Color tokenColor;
    if (contextTokens < numCtx * 0.7) {
      tokenColor = Colors.white38;
    } else if (contextTokens < numCtx * 0.9) {
      tokenColor = Colors.orange.shade300;
    } else {
      tokenColor = Colors.red.shade300;
    }

    return Container(
      color: const Color(0xE81E1F22),
      padding: const EdgeInsets.fromLTRB(24, 4, 24, 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                Text(
                  '~$contextTokens / $numCtx tokens',
                  style: TextStyle(color: tokenColor, fontSize: 10),
                ),
              ],
            ),
          ),
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Expanded(
                child: Shortcuts(
                  shortcuts: const {
                    SingleActivator(LogicalKeyboardKey.enter): _SendIntent(),
                    SingleActivator(LogicalKeyboardKey.enter, shift: true):
                        _NewlineIntent(),
                  },
                  child: Actions(
                    actions: {
                      _SendIntent: CallbackAction<_SendIntent>(
                        onInvoke: (intent) {
                          if (!ctrl.streaming) onSend();
                          return null;
                        },
                      ),
                      _NewlineIntent: CallbackAction<_NewlineIntent>(
                        onInvoke: (intent) {
                          final sel = controller.selection;
                          final base = controller.text;
                          final newText =
                              base.replaceRange(sel.start, sel.end, '\n');
                          controller.value = TextEditingValue(
                            text: newText,
                            selection: TextSelection.collapsed(
                              offset: sel.start + 1,
                            ),
                          );
                          return null;
                        },
                      ),
                    },
                    child: TextField(
                      controller: controller,
                      focusNode: focusNode,
                      maxLines: 8,
                      minLines: 1,
                      autofocus: true,
                      textInputAction: TextInputAction.newline,
                      decoration: const InputDecoration(
                        hintText:
                            'Message  (Enter = send, Shift+Enter = newline, drop files to attach)',
                        contentPadding: EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 14,
                        ),
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(
                height: 48,
                child: FilledButton(
                  onPressed: ctrl.streaming ? null : onSend,
                  style: FilledButton.styleFrom(
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                  child: const Icon(Icons.send_rounded, size: 20),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _SendIntent extends Intent {
  const _SendIntent();
}

class _NewlineIntent extends Intent {
  const _NewlineIntent();
}
