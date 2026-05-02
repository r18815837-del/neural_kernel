import 'dart:convert';
import 'dart:math' as math;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_highlight/flutter_highlight.dart';
import 'package:flutter_highlight/themes/atom-one-dark.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:flutter_math_fork/flutter_math.dart';
import 'package:provider/provider.dart';

import 'chat_controller.dart';
import 'main.dart';
import 'models.dart';
import 'tools.dart';

// ════════════════════════════════════════════════════════════════════════
// Sidebar
// ════════════════════════════════════════════════════════════════════════
class SidebarPanel extends StatefulWidget {
  const SidebarPanel({super.key, required this.width});
  final double width;

  @override
  State<SidebarPanel> createState() => _SidebarPanelState();
}

class _SidebarPanelState extends State<SidebarPanel> {
  final _searchCtrl = TextEditingController();
  final _searchFocus = FocusNode();

  @override
  void initState() {
    super.initState();
    SearchFocus.node = _searchFocus;
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    _searchFocus.dispose();
    if (identical(SearchFocus.node, _searchFocus)) {
      SearchFocus.node = null;
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ctrl = context.watch<ChatController>();
    final active = ctrl.activeConversation?.id;

    return SizedBox(
      width: widget.width,
      child: Container(
        color: const Color(0xE817181A),
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.fromLTRB(16, 18, 16, 8),
              alignment: Alignment.centerLeft,
              child: const Row(
                children: [
                  Icon(Icons.hub_outlined, size: 18, color: Color(0xFF7C9CFF)),
                  SizedBox(width: 8),
                  Text(
                    'NK Chat',
                    style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                      color: Colors.white,
                    ),
                  ),
                ],
              ),
            ),
            // New chat
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 4, 12, 8),
              child: SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  icon: const Icon(Icons.add, size: 18),
                  label: const Text('New chat  ·  Ctrl+N'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.white,
                    side: const BorderSide(color: Colors.white24),
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  onPressed: () => ctrl.createConversation(),
                ),
              ),
            ),
            // Search
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 0, 12, 8),
              child: TextField(
                controller: _searchCtrl,
                focusNode: _searchFocus,
                style: const TextStyle(color: Colors.white, fontSize: 12),
                decoration: InputDecoration(
                  hintText: 'Search  ·  Ctrl+K',
                  hintStyle:
                      const TextStyle(color: Colors.white38, fontSize: 12),
                  prefixIcon: const Icon(
                    Icons.search,
                    color: Colors.white38,
                    size: 16,
                  ),
                  prefixIconConstraints:
                      const BoxConstraints(minWidth: 32, minHeight: 32),
                  isDense: true,
                  contentPadding:
                      const EdgeInsets.symmetric(vertical: 8, horizontal: 8),
                  suffixIcon: ctrl.searchQuery.isEmpty
                      ? null
                      : IconButton(
                          icon: const Icon(Icons.close,
                              color: Colors.white38, size: 14),
                          padding: EdgeInsets.zero,
                          constraints: const BoxConstraints(
                            minWidth: 28,
                            minHeight: 28,
                          ),
                          onPressed: () {
                            _searchCtrl.clear();
                            ctrl.searchQuery = '';
                          },
                        ),
                ),
                onChanged: (v) => ctrl.searchQuery = v,
              ),
            ),
            const Divider(height: 1),
            Expanded(
              child: ctrl.conversations.isEmpty
                  ? _SidebarEmpty(searching: _searchCtrl.text.isNotEmpty)
                  : _GroupedChatList(
                      conversations: ctrl.conversations,
                      activeId: active,
                      onSelect: ctrl.selectConversation,
                      onDelete: ctrl.deleteConversation,
                      onRename: ctrl.renameConversation,
                    ),
            ),
            const Divider(height: 1),
            Padding(
              padding: const EdgeInsets.all(12),
              child: _FooterStatus(),
            ),
          ],
        ),
      ),
    );
  }
}

class _GroupedChatList extends StatelessWidget {
  const _GroupedChatList({
    required this.conversations,
    required this.activeId,
    required this.onSelect,
    required this.onDelete,
    required this.onRename,
  });

  final List<Conversation> conversations;
  final String? activeId;
  final void Function(String id) onSelect;
  final void Function(String id) onDelete;
  final void Function(String id, String newTitle) onRename;

  Map<String, List<Conversation>> _groupByDate() {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final yesterday = today.subtract(const Duration(days: 1));
    final weekAgo = today.subtract(const Duration(days: 7));

    final groups = <String, List<Conversation>>{
      'Today': [],
      'Yesterday': [],
      'Last 7 days': [],
      'Older': [],
    };
    for (final c in conversations) {
      final dt = DateTime(c.updatedAt.year, c.updatedAt.month, c.updatedAt.day);
      if (dt.isAtSameMomentAs(today) || dt.isAfter(today)) {
        groups['Today']!.add(c);
      } else if (dt.isAtSameMomentAs(yesterday)) {
        groups['Yesterday']!.add(c);
      } else if (dt.isAfter(weekAgo)) {
        groups['Last 7 days']!.add(c);
      } else {
        groups['Older']!.add(c);
      }
    }
    return groups;
  }

  @override
  Widget build(BuildContext context) {
    final groups = _groupByDate();
    final nonEmpty = groups.entries.where((e) => e.value.isNotEmpty).toList();

    return ListView.builder(
      padding: const EdgeInsets.only(top: 6, bottom: 12),
      itemCount: nonEmpty.fold<int>(0, (acc, e) => acc + 1 + e.value.length),
      itemBuilder: (context, flatIndex) {
        var i = flatIndex;
        for (final entry in nonEmpty) {
          if (i == 0) return _GroupHeader(title: entry.key);
          i -= 1;
          if (i < entry.value.length) {
            final c = entry.value[i];
            return _ChatListItem(
              conversation: c,
              selected: c.id == activeId,
              onTap: () => onSelect(c.id),
              onDelete: () => onDelete(c.id),
              onRename: (newTitle) => onRename(c.id, newTitle),
            );
          }
          i -= entry.value.length;
        }
        return const SizedBox.shrink();
      },
    );
  }
}

class _GroupHeader extends StatelessWidget {
  const _GroupHeader({required this.title});
  final String title;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 14, 12, 4),
      child: Text(
        title.toUpperCase(),
        style: const TextStyle(
          color: Colors.white38,
          fontSize: 10,
          fontWeight: FontWeight.w700,
          letterSpacing: 1.1,
        ),
      ),
    );
  }
}

class _SidebarEmpty extends StatelessWidget {
  const _SidebarEmpty({this.searching = false});
  final bool searching;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Text(
          searching
              ? 'Нет совпадений.\nИзмени поисковый запрос.'
              : 'Нет чатов.\nНажми «New chat» сверху.',
          textAlign: TextAlign.center,
          style: const TextStyle(color: Colors.white38, fontSize: 12),
        ),
      ),
    );
  }
}

class _ChatListItem extends StatefulWidget {
  const _ChatListItem({
    required this.conversation,
    required this.selected,
    required this.onTap,
    required this.onDelete,
    required this.onRename,
  });

  final Conversation conversation;
  final bool selected;
  final VoidCallback onTap;
  final VoidCallback onDelete;
  final void Function(String newTitle) onRename;

  @override
  State<_ChatListItem> createState() => _ChatListItemState();
}

class _ChatListItemState extends State<_ChatListItem> {
  bool _hovered = false;
  bool _renaming = false;
  final _renameCtrl = TextEditingController();
  final _renameFocus = FocusNode();

  @override
  void dispose() {
    _renameCtrl.dispose();
    _renameFocus.dispose();
    super.dispose();
  }

  void _startRename() {
    _renameCtrl.text = widget.conversation.title;
    setState(() => _renaming = true);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _renameFocus.requestFocus();
      _renameCtrl.selection = TextSelection(
        baseOffset: 0,
        extentOffset: _renameCtrl.text.length,
      );
    });
  }

  void _commitRename() {
    widget.onRename(_renameCtrl.text);
    setState(() => _renaming = false);
  }

  @override
  Widget build(BuildContext context) {
    final bg = widget.selected
        ? const Color(0xFF2A2B2E)
        : _hovered
            ? const Color(0xFF1F2022)
            : Colors.transparent;

    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      cursor: SystemMouseCursors.click,
      child: GestureDetector(
        onTap: widget.onTap,
        onDoubleTap: _startRename,
        child: Container(
          margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
          padding: const EdgeInsets.fromLTRB(12, 10, 6, 10),
          decoration: BoxDecoration(
            color: bg,
            borderRadius: BorderRadius.circular(6),
          ),
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (_renaming)
                      TextField(
                        controller: _renameCtrl,
                        focusNode: _renameFocus,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 13,
                        ),
                        decoration: const InputDecoration(
                          isDense: true,
                          contentPadding: EdgeInsets.symmetric(vertical: 2),
                          fillColor: Colors.transparent,
                        ),
                        onSubmitted: (_) => _commitRename(),
                        onEditingComplete: _commitRename,
                        onTapOutside: (_) => _commitRename(),
                      )
                    else
                      Text(
                        widget.conversation.title,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 13,
                        ),
                      ),
                    const SizedBox(height: 2),
                    Text(
                      widget.conversation.model,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        color: Colors.white38,
                        fontSize: 11,
                      ),
                    ),
                  ],
                ),
              ),
              if ((_hovered || widget.selected) && !_renaming) ...[
                IconButton(
                  icon: const Icon(Icons.edit_outlined, size: 14),
                  color: Colors.white54,
                  tooltip: 'Rename  ·  double-click',
                  padding: EdgeInsets.zero,
                  constraints:
                      const BoxConstraints(minWidth: 26, minHeight: 26),
                  onPressed: _startRename,
                ),
                IconButton(
                  icon: const Icon(Icons.delete_outline, size: 14),
                  color: Colors.white54,
                  tooltip: 'Delete',
                  padding: EdgeInsets.zero,
                  constraints:
                      const BoxConstraints(minWidth: 26, minHeight: 26),
                  onPressed: widget.onDelete,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _FooterStatus extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final ctrl = context.watch<ChatController>();
    final ok = ctrl.availableModels.isNotEmpty;
    return Row(
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            color: ok ? Colors.green.shade400 : Colors.red.shade400,
            shape: BoxShape.circle,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            ok
                ? 'Ollama: ${ctrl.availableModels.length} models'
                : 'Ollama offline',
            style: const TextStyle(color: Colors.white54, fontSize: 11),
          ),
        ),
        IconButton(
          tooltip: 'Refresh',
          icon: const Icon(Icons.refresh, size: 16),
          color: Colors.white54,
          padding: EdgeInsets.zero,
          constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
          onPressed: () => ctrl.refreshModels(),
        ),
      ],
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// Model selector
// ════════════════════════════════════════════════════════════════════════
class ModelSelector extends StatelessWidget {
  const ModelSelector({super.key});

  @override
  Widget build(BuildContext context) {
    final ctrl = context.watch<ChatController>();
    final conv = ctrl.activeConversation;
    final models = ctrl.availableModels;

    if (models.isEmpty) {
      return const Tooltip(
        message: 'Нет чат-моделей. Установи: ollama pull qwen2.5-coder:7b',
        child: Text(
          'no chat models',
          style: TextStyle(color: Colors.orange, fontSize: 12),
        ),
      );
    }

    final current = conv?.model ?? ctrl.selectedModel;
    final value = models.contains(current) ? current : models.first;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10),
      decoration: BoxDecoration(
        color: const Color(0xFF26272A),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white12),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: value,
          dropdownColor: const Color(0xFF26272A),
          iconSize: 16,
          style: const TextStyle(color: Colors.white, fontSize: 12),
          items: models
              .map((m) => DropdownMenuItem(value: m, child: Text(m)))
              .toList(),
          onChanged: (v) {
            if (v == null) return;
            if (conv != null) {
              ctrl.updateActiveModel(v);
            } else {
              ctrl.selectedModel = v;
            }
          },
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// Code panel — правая панель с файлом / diff
// ════════════════════════════════════════════════════════════════════════
class CodePanel extends StatelessWidget {
  const CodePanel({super.key});

  @override
  Widget build(BuildContext context) {
    final ctrl = context.watch<ChatController>();
    final path = ctrl.panelFilePath;
    final content = ctrl.panelFileContent ?? '';
    final preview = ctrl.panelPreviewEdit;

    return Container(
      color: const Color(0xE817181A),
      child: Column(
        children: [
          // Header
          Container(
            padding: const EdgeInsets.fromLTRB(12, 8, 8, 8),
            decoration: const BoxDecoration(
              border: Border(bottom: BorderSide(color: Colors.white12)),
            ),
            child: Row(
              children: [
                const Icon(Icons.description_outlined,
                    size: 14, color: Colors.white54),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    path ?? 'Code panel (empty)',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: path == null ? Colors.white38 : Colors.white,
                      fontSize: 12,
                      fontFamily: 'Cascadia Mono',
                      fontFamilyFallback: const ['Consolas', 'monospace'],
                    ),
                  ),
                ),
                if (preview != null) ...[
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: const Color(0xFFE77E8A).withOpacity(0.2),
                      borderRadius: BorderRadius.circular(3),
                    ),
                    child: const Text(
                      'preview',
                      style: TextStyle(
                        color: Color(0xFFE77E8A),
                        fontSize: 10,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  const SizedBox(width: 4),
                ],
                if (path != null)
                  IconButton(
                    icon: const Icon(Icons.close, size: 14),
                    color: Colors.white54,
                    tooltip: 'Close',
                    padding: EdgeInsets.zero,
                    constraints:
                        const BoxConstraints(minWidth: 24, minHeight: 24),
                    onPressed: () => ctrl.clearPanel(),
                  ),
              ],
            ),
          ),
          // Body
          Expanded(
            child: path == null
                ? const _CodePanelEmpty()
                : _CodePanelContent(
                    path: path,
                    content: content,
                    preview: preview,
                  ),
          ),
        ],
      ),
    );
  }
}

class _CodePanelEmpty extends StatelessWidget {
  const _CodePanelEmpty();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 300),
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.code_off,
                size: 40,
                color: Colors.white24,
              ),
              const SizedBox(height: 12),
              const Text(
                'Нет открытого файла',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white54,
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 6),
              const Text(
                'В Agent mode модель может автоматически показывать '
                'файлы здесь, когда использует read_file или edit_file.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white38,
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

class _CodePanelContent extends StatelessWidget {
  const _CodePanelContent({
    required this.path,
    required this.content,
    required this.preview,
  });

  final String path;
  final String content;
  final ToolCall? preview;

  String _languageFromPath(String p) {
    final ext = p.contains('.') ? p.split('.').last.toLowerCase() : '';
    return switch (ext) {
      'py' => 'python',
      'js' => 'javascript',
      'ts' => 'typescript',
      'dart' => 'dart',
      'rs' => 'rust',
      'go' => 'go',
      'java' => 'java',
      'cpp' || 'c' || 'h' || 'hpp' => 'cpp',
      'html' => 'html',
      'css' => 'css',
      'json' => 'json',
      'yaml' || 'yml' => 'yaml',
      'md' => 'markdown',
      'sh' => 'bash',
      'ps1' => 'powershell',
      'sql' => 'sql',
      _ => 'plaintext',
    };
  }

  @override
  Widget build(BuildContext context) {
    // Режим diff preview для edit_file
    if (preview != null && preview!.name == 'edit_file') {
      final oldStr = preview!.arguments['old_string'] as String? ?? '';
      final newStr = preview!.arguments['new_string'] as String? ?? '';
      return _DiffView(
        fullContent: content,
        oldStr: oldStr,
        newStr: newStr,
      );
    }
    // Режим diff для write_file (просто новый контент)
    if (preview != null && preview!.name == 'write_file') {
      final newContent = preview!.arguments['content'] as String? ?? '';
      return Container(
        color: const Color(0xFF1B1C1E),
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(12),
          child: HighlightView(
            newContent,
            language: _languageFromPath(path),
            theme: atomOneDarkTheme,
            textStyle: const TextStyle(
              fontFamily: 'Cascadia Mono',
              fontFamilyFallback: ['Consolas', 'monospace'],
              fontSize: 12,
              height: 1.5,
            ),
          ),
        ),
      );
    }
    // Обычный просмотр
    return Container(
      color: const Color(0xFF1B1C1E),
      child: Scrollbar(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(12),
          child: HighlightView(
            content.isEmpty ? '// empty file' : content,
            language: _languageFromPath(path),
            theme: atomOneDarkTheme,
            textStyle: const TextStyle(
              fontFamily: 'Cascadia Mono',
              fontFamilyFallback: ['Consolas', 'monospace'],
              fontSize: 12,
              height: 1.5,
            ),
          ),
        ),
      ),
    );
  }
}

/// Показывает diff: старые строки с красным фоном, новые — с зелёным.
class _DiffView extends StatelessWidget {
  const _DiffView({
    required this.fullContent,
    required this.oldStr,
    required this.newStr,
  });

  final String fullContent;
  final String oldStr;
  final String newStr;

  @override
  Widget build(BuildContext context) {
    // Находим место old_string в full content, показываем с контекстом.
    final idx = fullContent.indexOf(oldStr);
    final contextBefore = 300;
    final contextAfter = 300;

    String beforeStr = '';
    String afterStr = '';
    if (idx >= 0) {
      final startCtx = (idx - contextBefore).clamp(0, fullContent.length);
      final endCtx = (idx + oldStr.length + contextAfter)
          .clamp(0, fullContent.length);
      beforeStr = fullContent.substring(startCtx, idx);
      afterStr = fullContent.substring(idx + oldStr.length, endCtx);
    }

    return Container(
      color: const Color(0xFF1B1C1E),
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(12),
        child: SelectableText.rich(
          TextSpan(
            style: const TextStyle(
              fontFamily: 'Cascadia Mono',
              fontFamilyFallback: ['Consolas', 'monospace'],
              fontSize: 12,
              height: 1.5,
              color: Color(0xFFB8BCC8),
            ),
            children: [
              if (idx < 0)
                const TextSpan(
                  text: '[old_string не найден в текущем content]\n\n',
                  style: TextStyle(color: Colors.orangeAccent),
                ),
              TextSpan(text: beforeStr),
              TextSpan(
                text: oldStr,
                style: TextStyle(
                  backgroundColor:
                      const Color(0xFFFF6868).withOpacity(0.15),
                  color: const Color(0xFFFFC5C5),
                  decoration: TextDecoration.lineThrough,
                ),
              ),
              TextSpan(
                text: newStr,
                style: TextStyle(
                  backgroundColor:
                      const Color(0xFF44BB66).withOpacity(0.15),
                  color: const Color(0xFFB8E8C6),
                ),
              ),
              TextSpan(text: afterStr),
            ],
          ),
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// Panel toggle buttons
// ════════════════════════════════════════════════════════════════════════
class PanelToggle extends StatelessWidget {
  const PanelToggle({
    super.key,
    required this.icon,
    required this.active,
    required this.tooltip,
    required this.onPressed,
  });

  final IconData icon;
  final bool active;
  final String tooltip;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: tooltip,
      child: IconButton(
        icon: Icon(icon, size: 16),
        color: active ? const Color(0xFF7C9CFF) : Colors.white54,
        padding: EdgeInsets.zero,
        constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
        onPressed: onPressed,
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// Agent mode chip (в шапке)
// ════════════════════════════════════════════════════════════════════════
class AgentToggle extends StatelessWidget {
  const AgentToggle({super.key});

  @override
  Widget build(BuildContext context) {
    final ctrl = context.watch<ChatController>();
    final on = ctrl.agentMode;
    final hasWorkspace = ctrl.workspaceRoot.isNotEmpty;

    final disabledReason =
        !hasWorkspace ? 'Выбери workspace в настройках' : null;

    return Tooltip(
      message: disabledReason ??
          (on
              ? 'Agent mode ON — модель может читать/править файлы '
                  'в ${ctrl.workspaceRoot}'
              : 'Agent mode OFF'),
      child: InkWell(
        onTap: () {
          if (!hasWorkspace) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text(
                    'Сначала выбери workspace в настройках (шестерёнка → Workspace)'),
                behavior: SnackBarBehavior.floating,
                duration: Duration(seconds: 3),
              ),
            );
            return;
          }
          ctrl.agentMode = !on;
        },
        borderRadius: BorderRadius.circular(8),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: on ? const Color(0xFF553A3F) : const Color(0xFF26272A),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: on
                  ? const Color(0xFFE77E8A)
                  : (hasWorkspace ? Colors.white12 : Colors.white10),
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                on ? Icons.smart_toy : Icons.smart_toy_outlined,
                size: 14,
                color: on
                    ? const Color(0xFFE77E8A)
                    : (hasWorkspace ? Colors.white54 : Colors.white24),
              ),
              const SizedBox(width: 6),
              Text(
                'Agent',
                style: TextStyle(
                  fontSize: 11,
                  color: on
                      ? Colors.white
                      : (hasWorkspace ? Colors.white70 : Colors.white38),
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// RAG toggle chip (в шапке рядом с model selector)
// ════════════════════════════════════════════════════════════════════════
class RagToggle extends StatelessWidget {
  const RagToggle({super.key});

  @override
  Widget build(BuildContext context) {
    final ctrl = context.watch<ChatController>();
    final on = ctrl.useCodebaseContext;
    return Tooltip(
      message: on
          ? 'RAG включён — модель видит релевантные куски кода из Qdrant'
          : 'RAG выключен',
      child: InkWell(
        onTap: () => ctrl.useCodebaseContext = !on,
        borderRadius: BorderRadius.circular(8),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: on
                ? const Color(0xFF2C3D55)
                : const Color(0xFF26272A),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: on ? const Color(0xFF7C9CFF) : Colors.white12,
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                on ? Icons.auto_awesome : Icons.auto_awesome_outlined,
                size: 14,
                color: on ? const Color(0xFF7C9CFF) : Colors.white54,
              ),
              const SizedBox(width: 6),
              Text(
                'Codebase RAG',
                style: TextStyle(
                  fontSize: 11,
                  color: on ? Colors.white : Colors.white70,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// Message bubble
// ════════════════════════════════════════════════════════════════════════
class MessageBubble extends StatefulWidget {
  const MessageBubble({
    super.key,
    required this.message,
    this.highlightQuery = '',
  });
  final Message message;
  final String highlightQuery;

  @override
  State<MessageBubble> createState() => _MessageBubbleState();
}

class _MessageBubbleState extends State<MessageBubble> {
  bool _hovered = false;
  bool _editing = false;
  final _editCtrl = TextEditingController();

  @override
  void dispose() {
    _editCtrl.dispose();
    super.dispose();
  }

  void _startEdit() {
    _editCtrl.text = widget.message.content;
    setState(() => _editing = true);
  }

  void _submitEdit() {
    final ctrl = context.read<ChatController>();
    final newText = _editCtrl.text.trim();
    setState(() => _editing = false);
    if (newText.isEmpty || newText == widget.message.content) return;
    ctrl.editUserMessage(widget.message.id, newText);
  }

  @override
  Widget build(BuildContext context) {
    final message = widget.message;
    // tool-role messages рендерим отдельным минимальным виджетом.
    if (message.role == MessageRole.tool) {
      return _ToolResultCard(message: message);
    }
    final isUser = message.role == MessageRole.user;
    final ctrl = context.read<ChatController>();

    final isLastAssistant = !isUser &&
        ctrl.activeConversation?.messages.isNotEmpty == true &&
        ctrl.activeConversation?.messages.last.id == message.id;

    final bubble = MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _Avatar(isUser: isUser),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(
                        isUser ? 'You' : 'Assistant',
                        style: const TextStyle(
                          fontWeight: FontWeight.w600,
                          color: Colors.white,
                          fontSize: 13,
                        ),
                      ),
                      const SizedBox(width: 10),
                      if (message.streaming)
                        const TypingIndicator(
                          color: Color(0xFF9AA1B0),
                          dotSize: 5,
                          spacing: 4,
                        ),
                      // timestamp — тонкий, только на hover
                      if (_hovered && !message.streaming) ...[
                        const SizedBox(width: 10),
                        Text(
                          _formatTimestamp(message.createdAt),
                          style: const TextStyle(
                            color: Colors.white30,
                            fontSize: 11,
                          ),
                        ),
                      ],
                      const Spacer(),
                      if (!message.streaming && message.content.isNotEmpty)
                        AnimatedOpacity(
                          opacity: _hovered ? 1.0 : 0.0,
                          duration: const Duration(milliseconds: 120),
                          child: _MessageActionsRow(
                            message: message,
                            isUser: isUser,
                            canRegenerate: isLastAssistant && !ctrl.streaming,
                            onRegenerate: () => ctrl.regenerateLastAssistant(),
                            onDelete: () => ctrl.deleteMessage(message.id),
                            onEdit: isUser ? _startEdit : null,
                          ),
                        ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  if (_editing && isUser)
                    _EditArea(
                      controller: _editCtrl,
                      onSubmit: _submitEdit,
                      onCancel: () => setState(() => _editing = false),
                    )
                  else
                    _MessageBody(
                      message: message,
                      highlightQuery: widget.highlightQuery,
                    ),
                  // Tool calls (agent mode)
                  if (!isUser &&
                      message.toolCalls != null &&
                      message.toolCalls!.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: _ToolCallsList(calls: message.toolCalls!),
                    ),
                  // RAG-ссылки (если были)
                  if (!isUser &&
                      !message.streaming &&
                      message.ragRefs != null &&
                      message.ragRefs!.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: _RagChunksCard(refs: message.ragRefs!),
                    ),
                  // Quick actions — только для последнего assistant-ответа
                  if (!isUser &&
                      !message.streaming &&
                      !ctrl.streaming &&
                      message.content.isNotEmpty &&
                      isLastAssistant)
                    Padding(
                      padding: const EdgeInsets.only(top: 10),
                      child: _QuickActions(
                        onFollowUp: (prompt) => ctrl.sendFollowUp(prompt),
                      ),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );

    return TweenAnimationBuilder<double>(
      tween: Tween<double>(begin: 0, end: 1),
      duration: const Duration(milliseconds: 240),
      curve: Curves.easeOut,
      builder: (context, value, child) {
        return Opacity(
          opacity: value,
          child: Transform.translate(
            offset: Offset(0, (1 - value) * 8),
            child: child,
          ),
        );
      },
      child: bubble,
    );
  }

  String _formatTimestamp(DateTime t) {
    final now = DateTime.now();
    final diff = now.difference(t);
    if (diff.inSeconds < 60) return 'just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    if (diff.inDays < 7) return '${diff.inDays}d ago';
    return '${t.year}-${t.month.toString().padLeft(2, '0')}-'
        '${t.day.toString().padLeft(2, '0')}';
  }
}

class _EditArea extends StatelessWidget {
  const _EditArea({
    required this.controller,
    required this.onSubmit,
    required this.onCancel,
  });

  final TextEditingController controller;
  final VoidCallback onSubmit;
  final VoidCallback onCancel;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xFF1B1C1E),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: const Color(0xFF7C9CFF)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          TextField(
            controller: controller,
            maxLines: 10,
            minLines: 2,
            autofocus: true,
            style: const TextStyle(color: Colors.white, fontSize: 14),
            decoration: const InputDecoration(
              filled: false,
              border: InputBorder.none,
              contentPadding: EdgeInsets.zero,
              hintText: 'Edit your message…',
            ),
          ),
          const SizedBox(height: 6),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextButton(onPressed: onCancel, child: const Text('Cancel')),
              const SizedBox(width: 8),
              FilledButton(
                onPressed: onSubmit,
                child: const Text('Send & Regenerate'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _MessageActionsRow extends StatelessWidget {
  const _MessageActionsRow({
    required this.message,
    required this.isUser,
    required this.canRegenerate,
    required this.onRegenerate,
    required this.onDelete,
    this.onEdit,
  });

  final Message message;
  final bool isUser;
  final bool canRegenerate;
  final VoidCallback onRegenerate;
  final VoidCallback onDelete;
  final VoidCallback? onEdit;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        _MiniIconButton(
          icon: Icons.content_copy_outlined,
          tooltip: 'Copy',
          onPressed: () async {
            await Clipboard.setData(ClipboardData(text: message.content));
            if (context.mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Copied'),
                  duration: Duration(seconds: 1),
                  behavior: SnackBarBehavior.floating,
                ),
              );
            }
          },
        ),
        if (isUser && onEdit != null)
          _MiniIconButton(
            icon: Icons.edit_outlined,
            tooltip: 'Edit & regenerate',
            onPressed: onEdit,
          ),
        if (!isUser)
          _MiniIconButton(
            icon: Icons.refresh,
            tooltip:
                canRegenerate ? 'Regenerate' : 'Только для последнего ответа',
            onPressed: canRegenerate ? onRegenerate : null,
          ),
        _MiniIconButton(
          icon: Icons.delete_outline,
          tooltip: 'Delete',
          onPressed: onDelete,
        ),
      ],
    );
  }
}

class _MiniIconButton extends StatelessWidget {
  const _MiniIconButton({
    required this.icon,
    required this.tooltip,
    required this.onPressed,
  });

  final IconData icon;
  final String tooltip;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    final disabled = onPressed == null;
    return Tooltip(
      message: tooltip,
      child: IconButton(
        icon: Icon(icon, size: 15),
        color: disabled ? Colors.white24 : Colors.white54,
        hoverColor: Colors.white10,
        padding: EdgeInsets.zero,
        constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
        onPressed: onPressed,
      ),
    );
  }
}

class _Avatar extends StatelessWidget {
  const _Avatar({required this.isUser});
  final bool isUser;

  @override
  Widget build(BuildContext context) {
    final gradient = isUser
        ? const LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF8B7FFF), Color(0xFF5B9BFF)],
          )
        : const LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF6FE0C5), Color(0xFF4AB2D1)],
          );

    return Container(
      width: 30,
      height: 30,
      decoration: BoxDecoration(
        gradient: gradient,
        borderRadius: BorderRadius.circular(7),
        boxShadow: [
          BoxShadow(
            color: (isUser
                    ? const Color(0xFF5B9BFF)
                    : const Color(0xFF4AB2D1))
                .withOpacity(0.25),
            blurRadius: 6,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      alignment: Alignment.center,
      child: Text(
        isUser ? 'U' : 'A',
        style: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.w700,
          fontSize: 13,
          letterSpacing: 0.3,
        ),
      ),
    );
  }
}

class _MessageBody extends StatelessWidget {
  const _MessageBody({
    required this.message,
    this.highlightQuery = '',
  });
  final Message message;
  final String highlightQuery;

  @override
  Widget build(BuildContext context) {
    if (message.role == MessageRole.user) {
      return _HighlightableSelectable(
        text: message.content,
        query: highlightQuery,
        style: const TextStyle(
          color: Color(0xFFE6E7E9),
          fontSize: 14,
          height: 1.55,
        ),
      );
    }

    if (message.content.isEmpty && message.streaming) {
      return const Padding(
        padding: EdgeInsets.symmetric(vertical: 6),
        child: TypingIndicator(
          color: Color(0xFF9AA1B0),
          dotSize: 6,
          spacing: 5,
        ),
      );
    }

    return _MarkdownWithMath(source: message.content);
  }
}

/// Текст с подсветкой совпадений (для in-chat search).
class _HighlightableSelectable extends StatelessWidget {
  const _HighlightableSelectable({
    required this.text,
    required this.query,
    required this.style,
  });

  final String text;
  final String query;
  final TextStyle style;

  @override
  Widget build(BuildContext context) {
    if (query.trim().isEmpty) {
      return SelectableText(text, style: style);
    }

    final q = query.toLowerCase();
    final lower = text.toLowerCase();
    final spans = <TextSpan>[];
    int cursor = 0;
    while (true) {
      final i = lower.indexOf(q, cursor);
      if (i < 0) {
        if (cursor < text.length) {
          spans.add(TextSpan(text: text.substring(cursor), style: style));
        }
        break;
      }
      if (i > cursor) {
        spans.add(TextSpan(text: text.substring(cursor, i), style: style));
      }
      spans.add(
        TextSpan(
          text: text.substring(i, i + q.length),
          style: style.copyWith(
            backgroundColor: const Color(0xFFFFE066).withOpacity(0.35),
            color: Colors.white,
            fontWeight: FontWeight.w600,
          ),
        ),
      );
      cursor = i + q.length;
    }

    return SelectableText.rich(TextSpan(children: spans));
  }
}

// ════════════════════════════════════════════════════════════════════════
// RAG chunks card — expandable под assistant-ответом
// ════════════════════════════════════════════════════════════════════════
class _RagChunksCard extends StatefulWidget {
  const _RagChunksCard({required this.refs});
  final List<MessageRagRef> refs;

  @override
  State<_RagChunksCard> createState() => _RagChunksCardState();
}

class _RagChunksCardState extends State<_RagChunksCard> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF1B1C1E),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          InkWell(
            onTap: () => setState(() => _expanded = !_expanded),
            borderRadius: BorderRadius.circular(6),
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 8, 8, 8),
              child: Row(
                children: [
                  const Icon(
                    Icons.auto_awesome,
                    size: 14,
                    color: Color(0xFF7C9CFF),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    'Used context · ${widget.refs.length} chunk${widget.refs.length == 1 ? '' : 's'}',
                    style: const TextStyle(
                      color: Colors.white70,
                      fontSize: 11,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const Spacer(),
                  Icon(
                    _expanded
                        ? Icons.keyboard_arrow_up
                        : Icons.keyboard_arrow_down,
                    size: 16,
                    color: Colors.white54,
                  ),
                ],
              ),
            ),
          ),
          if (_expanded)
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 0, 12, 10),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: widget.refs
                    .map((r) => _RagRefRow(ref: r))
                    .toList(),
              ),
            ),
        ],
      ),
    );
  }
}

class _RagRefRow extends StatelessWidget {
  const _RagRefRow({required this.ref});
  final MessageRagRef ref;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
            decoration: BoxDecoration(
              color: _kindColor(ref.kind).withOpacity(0.2),
              borderRadius: BorderRadius.circular(3),
            ),
            child: Text(
              ref.kind,
              style: TextStyle(
                color: _kindColor(ref.kind),
                fontSize: 10,
                fontFamily: 'Cascadia Mono',
                fontFamilyFallback: const ['Consolas', 'monospace'],
              ),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              '${ref.file}:${ref.startLine}  ·  ${ref.name}',
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                color: Color(0xFFE6E7E9),
                fontSize: 12,
                fontFamily: 'Cascadia Mono',
                fontFamilyFallback: ['Consolas', 'monospace'],
              ),
            ),
          ),
          const SizedBox(width: 8),
          Text(
            ref.score.toStringAsFixed(3),
            style: const TextStyle(
              color: Colors.white38,
              fontSize: 11,
            ),
          ),
        ],
      ),
    );
  }

  Color _kindColor(String kind) {
    return switch (kind) {
      'function' => const Color(0xFF6FE0C5),
      'class' => const Color(0xFFFFD86B),
      'module' => const Color(0xFF7C9CFF),
      'paragraph' => const Color(0xFFE19060),
      _ => const Color(0xFF9AA1B0),
    };
  }
}

// ════════════════════════════════════════════════════════════════════════
// Quick actions — Shorter / Simpler / Translate
// ════════════════════════════════════════════════════════════════════════
class _QuickActions extends StatelessWidget {
  const _QuickActions({required this.onFollowUp});

  final void Function(String prompt) onFollowUp;

  static const _actions = <({String label, IconData icon, String prompt})>[
    (
      label: 'Shorter',
      icon: Icons.compress,
      prompt:
          'Сделай предыдущий ответ короче. Сохрани суть и примеры кода, '
              'убери повторы и воду.',
    ),
    (
      label: 'Simpler',
      icon: Icons.auto_stories,
      prompt:
          'Переформулируй предыдущий ответ проще, как для новичка. Используй '
              'простые слова и аналогии, если уместно.',
    ),
    (
      label: 'Expand',
      icon: Icons.unfold_more,
      prompt:
          'Расширь предыдущий ответ: добавь детали, примеры, edge cases, '
              'указатели на подводные камни.',
    ),
    (
      label: 'Translate → EN',
      icon: Icons.translate,
      prompt: 'Translate the previous answer into English. Keep code as-is.',
    ),
    (
      label: 'Translate → RU',
      icon: Icons.translate,
      prompt: 'Переведи предыдущий ответ на русский. Код не трогай.',
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 6,
      runSpacing: 6,
      children: _actions
          .map(
            (a) => _QuickActionChip(
              label: a.label,
              icon: a.icon,
              onTap: () => onFollowUp(a.prompt),
            ),
          )
          .toList(),
    );
  }
}

class _QuickActionChip extends StatefulWidget {
  const _QuickActionChip({
    required this.label,
    required this.icon,
    required this.onTap,
  });

  final String label;
  final IconData icon;
  final VoidCallback onTap;

  @override
  State<_QuickActionChip> createState() => _QuickActionChipState();
}

class _QuickActionChipState extends State<_QuickActionChip> {
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
          duration: const Duration(milliseconds: 100),
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(
            color: _hovered
                ? const Color(0xFF2F3134)
                : const Color(0xFF222426),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(
              color: _hovered ? Colors.white24 : Colors.white12,
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(widget.icon, size: 12, color: Colors.white70),
              const SizedBox(width: 5),
              Text(
                widget.label,
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 11,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// Markdown + inline LaTeX рендерер
// ════════════════════════════════════════════════════════════════════════
class _MarkdownWithMath extends StatelessWidget {
  const _MarkdownWithMath({required this.source});
  final String source;

  @override
  Widget build(BuildContext context) {
    final parts = _splitMath(source);
    if (parts.length == 1 && !parts.first.isMath) {
      return _baseMarkdown(source);
    }

    final children = <Widget>[];
    for (final p in parts) {
      if (p.isMath) {
        children.add(
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Math.tex(
              p.text,
              textStyle: const TextStyle(color: Color(0xFFE6E7E9), fontSize: 14),
              onErrorFallback: (err) => _baseMarkdown(
                p.block ? '\$\$${p.text}\$\$' : '\$${p.text}\$',
              ),
            ),
          ),
        );
      } else if (p.text.trim().isNotEmpty) {
        children.add(_baseMarkdown(p.text));
      }
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: children,
    );
  }

  Widget _baseMarkdown(String src) {
    return MarkdownBody(
      data: src,
      selectable: true,
      styleSheet: _markdownStyle(),
      builders: {'code': _HighlightCodeBuilder()},
    );
  }

  /// Очень простой раздельщик — ищет $$...$$ и $...$, всё остальное
  /// как текст. Избегаем жадных матчей.
  List<_Piece> _splitMath(String src) {
    final result = <_Piece>[];
    final block = RegExp(r'\$\$([^$]+)\$\$', multiLine: true);
    final inline = RegExp(r'(?<!\$)\$([^$\n]+)\$(?!\$)');

    // Сначала выделяем $$…$$, потом внутри оставшегося — $…$.
    int cursor = 0;
    for (final m in block.allMatches(src)) {
      if (m.start > cursor) {
        result.addAll(_splitInline(src.substring(cursor, m.start), inline));
      }
      result.add(_Piece(text: m.group(1)!, isMath: true, block: true));
      cursor = m.end;
    }
    if (cursor < src.length) {
      result.addAll(_splitInline(src.substring(cursor), inline));
    }
    return result;
  }

  List<_Piece> _splitInline(String src, RegExp rx) {
    final out = <_Piece>[];
    int cursor = 0;
    for (final m in rx.allMatches(src)) {
      if (m.start > cursor) {
        out.add(_Piece(text: src.substring(cursor, m.start), isMath: false));
      }
      out.add(_Piece(text: m.group(1)!, isMath: true));
      cursor = m.end;
    }
    if (cursor < src.length) {
      out.add(_Piece(text: src.substring(cursor), isMath: false));
    }
    return out;
  }
}

class _Piece {
  _Piece({required this.text, required this.isMath, this.block = false});
  final String text;
  final bool isMath;
  final bool block;
}

MarkdownStyleSheet _markdownStyle() {
  return MarkdownStyleSheet(
    p: const TextStyle(color: Color(0xFFE6E7E9), fontSize: 14, height: 1.55),
    strong: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700),
    em: const TextStyle(fontStyle: FontStyle.italic, color: Color(0xFFE6E7E9)),
    a: const TextStyle(color: Color(0xFF7C9CFF)),
    code: const TextStyle(
      fontFamily: 'Cascadia Mono',
      fontFamilyFallback: ['Consolas', 'monospace'],
      fontSize: 13,
      color: Color(0xFFE6E7E9),
      backgroundColor: Color(0xFF2A2B2E),
    ),
    codeblockDecoration: BoxDecoration(
      color: const Color(0xFF1B1C1E),
      borderRadius: BorderRadius.circular(6),
      border: Border.all(color: Colors.white12),
    ),
    codeblockPadding: const EdgeInsets.all(12),
    blockquote: const TextStyle(color: Colors.white70),
    blockquoteDecoration: BoxDecoration(
      color: const Color(0xFF1B1C1E),
      border: const Border(
        left: BorderSide(color: Color(0xFF7C9CFF), width: 3),
      ),
      borderRadius: BorderRadius.circular(4),
    ),
    listBullet: const TextStyle(color: Color(0xFFE6E7E9), fontSize: 14),
    h1: const TextStyle(
        color: Colors.white, fontSize: 20, fontWeight: FontWeight.w700),
    h2: const TextStyle(
        color: Colors.white, fontSize: 18, fontWeight: FontWeight.w700),
    h3: const TextStyle(
        color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700),
    tableHead:
        const TextStyle(fontWeight: FontWeight.w700, color: Colors.white),
    tableBody: const TextStyle(color: Color(0xFFE6E7E9)),
  );
}

// ════════════════════════════════════════════════════════════════════════
// Code block renderer: line numbers + diff highlighting
// ════════════════════════════════════════════════════════════════════════
class _HighlightCodeBuilder extends MarkdownElementBuilder {
  @override
  Widget? visitElementAfter(element, TextStyle? preferredStyle) {
    final text = element.textContent;
    if (!text.contains('\n')) return null;

    String lang = '';
    try {
      final cls = element.attributes['class'] as String? ?? '';
      if (cls.startsWith('language-')) lang = cls.substring(9);
    } catch (_) {}

    return _CodeBlock(code: text, language: lang);
  }
}

class _CodeBlock extends StatelessWidget {
  const _CodeBlock({required this.code, required this.language});

  final String code;
  final String language;

  @override
  Widget build(BuildContext context) {
    final trimmed = code.trimRight();
    final isDiff = language == 'diff';
    final lines = trimmed.split('\n');
    final lineNumWidth = (lines.length.toString().length * 8.0) + 12;

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.symmetric(vertical: 6),
      decoration: BoxDecoration(
        color: const Color(0xFF1B1C1E),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _CodeHeader(language: language, code: trimmed),
          isDiff
              ? _DiffBody(lines: lines, lineNumWidth: lineNumWidth)
              : _HighlightBody(
                  lines: lines,
                  lineNumWidth: lineNumWidth,
                  language: language,
                ),
        ],
      ),
    );
  }
}

class _HighlightBody extends StatelessWidget {
  const _HighlightBody({
    required this.lines,
    required this.lineNumWidth,
    required this.language,
  });

  final List<String> lines;
  final double lineNumWidth;
  final String language;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Gutter с номерами строк.
          SizedBox(
            width: lineNumWidth,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: List.generate(lines.length, (i) {
                return SizedBox(
                  height: 19,
                  child: Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: Text(
                      '${i + 1}',
                      style: const TextStyle(
                        fontFamily: 'Cascadia Mono',
                        fontFamilyFallback: ['Consolas', 'monospace'],
                        color: Colors.white24,
                        fontSize: 12,
                        height: 1.5,
                      ),
                    ),
                  ),
                );
              }),
            ),
          ),
          // Сам подсвеченный код.
          Expanded(
            child: HighlightView(
              lines.join('\n'),
              language: language.isEmpty ? 'plaintext' : language,
              theme: atomOneDarkTheme,
              padding: const EdgeInsets.only(right: 12),
              textStyle: const TextStyle(
                fontFamily: 'Cascadia Mono',
                fontFamilyFallback: ['Consolas', 'monospace'],
                fontSize: 12.5,
                height: 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _DiffBody extends StatelessWidget {
  const _DiffBody({required this.lines, required this.lineNumWidth});

  final List<String> lines;
  final double lineNumWidth;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: List.generate(lines.length, (i) {
          final line = lines[i];
          Color? bg;
          Color fg = const Color(0xFFE6E7E9);
          if (line.startsWith('+') && !line.startsWith('+++')) {
            bg = const Color(0x2044BB66); // semi-green
            fg = const Color(0xFFB8E8C6);
          } else if (line.startsWith('-') && !line.startsWith('---')) {
            bg = const Color(0x20FF6868); // semi-red
            fg = const Color(0xFFFFC5C5);
          } else if (line.startsWith('@@')) {
            fg = const Color(0xFF9AA1B0);
          }

          return Container(
            color: bg,
            padding: const EdgeInsets.symmetric(vertical: 1),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SizedBox(
                  width: lineNumWidth,
                  child: Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: Text(
                      '${i + 1}',
                      textAlign: TextAlign.right,
                      style: const TextStyle(
                        fontFamily: 'Cascadia Mono',
                        fontFamilyFallback: ['Consolas', 'monospace'],
                        color: Colors.white24,
                        fontSize: 12,
                        height: 1.5,
                      ),
                    ),
                  ),
                ),
                Expanded(
                  child: SelectableText(
                    line.isEmpty ? ' ' : line,
                    style: TextStyle(
                      fontFamily: 'Cascadia Mono',
                      fontFamilyFallback: const ['Consolas', 'monospace'],
                      color: fg,
                      fontSize: 12.5,
                      height: 1.5,
                    ),
                  ),
                ),
              ],
            ),
          );
        }),
      ),
    );
  }
}

class _CodeHeader extends StatelessWidget {
  const _CodeHeader({required this.language, required this.code});
  final String language;
  final String code;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 6, 6, 6),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: Colors.white10)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
            decoration: BoxDecoration(
              color: _langColor(language).withOpacity(0.15),
              borderRadius: BorderRadius.circular(3),
            ),
            child: Text(
              language.isEmpty ? 'code' : language,
              style: TextStyle(
                color: _langColor(language),
                fontSize: 11,
                fontWeight: FontWeight.w600,
                fontFamily: 'Cascadia Mono',
                fontFamilyFallback: const ['Consolas', 'monospace'],
              ),
            ),
          ),
          const Spacer(),
          _CopyButton(text: code),
        ],
      ),
    );
  }

  static Color _langColor(String lang) {
    return switch (lang.toLowerCase()) {
      'python' || 'py' => const Color(0xFFFFD86B),
      'javascript' || 'js' || 'typescript' || 'ts' => const Color(0xFFF7DF1E),
      'rust' || 'rs' => const Color(0xFFFFA657),
      'dart' => const Color(0xFF4AB2D1),
      'bash' || 'sh' || 'shell' || 'powershell' => const Color(0xFF89D185),
      'diff' => const Color(0xFFE19060),
      'json' => const Color(0xFFB5CEA8),
      'yaml' || 'yml' => const Color(0xFFC586C0),
      _ => const Color(0xFF9AA1B0),
    };
  }
}

class _CopyButton extends StatefulWidget {
  const _CopyButton({required this.text});
  final String text;

  @override
  State<_CopyButton> createState() => _CopyButtonState();
}

class _CopyButtonState extends State<_CopyButton> {
  bool _copied = false;

  @override
  Widget build(BuildContext context) {
    return TextButton.icon(
      onPressed: () async {
        await Clipboard.setData(ClipboardData(text: widget.text));
        if (!mounted) return;
        setState(() => _copied = true);
        Future.delayed(const Duration(seconds: 2), () {
          if (mounted) setState(() => _copied = false);
        });
      },
      icon: Icon(
        _copied ? Icons.check : Icons.copy,
        size: 14,
        color: _copied ? Colors.green.shade300 : Colors.white54,
      ),
      label: Text(
        _copied ? 'copied' : 'copy',
        style: TextStyle(
          fontSize: 11,
          color: _copied ? Colors.green.shade300 : Colors.white54,
        ),
      ),
      style: TextButton.styleFrom(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
        minimumSize: Size.zero,
        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// Typing indicator
// ════════════════════════════════════════════════════════════════════════
class TypingIndicator extends StatefulWidget {
  const TypingIndicator({
    super.key,
    this.color = Colors.white70,
    this.dotSize = 5.0,
    this.spacing = 4.0,
    this.dotCount = 3,
    this.durationMs = 1100,
  });

  final Color color;
  final double dotSize;
  final double spacing;
  final int dotCount;
  final int durationMs;

  @override
  State<TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<TypingIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: Duration(milliseconds: widget.durationMs),
    )..repeat();
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  double _wave(double t) => math.sin(t.clamp(0.0, 1.0) * math.pi);

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _ctrl,
      builder: (context, _) {
        return Row(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: List.generate(widget.dotCount, (i) {
            final offset = i / widget.dotCount * 0.6;
            final phase = (_ctrl.value - offset) % 1.0;
            final wave = _wave(phase);
            final opacity = 0.3 + 0.7 * wave;
            final scale = 0.85 + 0.25 * wave;
            return Padding(
              padding: EdgeInsets.only(
                right: i == widget.dotCount - 1 ? 0 : widget.spacing,
              ),
              child: Transform.scale(
                scale: scale,
                child: Opacity(
                  opacity: opacity,
                  child: Container(
                    width: widget.dotSize,
                    height: widget.dotSize,
                    decoration: BoxDecoration(
                      color: widget.color,
                      shape: BoxShape.circle,
                    ),
                  ),
                ),
              ),
            );
          }),
        );
      },
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// Tool call visualization (agent mode)
// ════════════════════════════════════════════════════════════════════════
class _ToolCallsList extends StatelessWidget {
  const _ToolCallsList({required this.calls});
  final List<ToolCall> calls;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: calls.map((c) => _ToolCallCard(call: c)).toList(),
    );
  }
}

class _ToolCallCard extends StatefulWidget {
  const _ToolCallCard({required this.call});
  final ToolCall call;

  @override
  State<_ToolCallCard> createState() => _ToolCallCardState();
}

class _ToolCallCardState extends State<_ToolCallCard> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final args = widget.call.arguments;
    final summary = _summarize(widget.call.name, args);

    return Container(
      margin: const EdgeInsets.only(bottom: 4),
      decoration: BoxDecoration(
        color: const Color(0xFF1B1C1E),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          InkWell(
            onTap: () => setState(() => _expanded = !_expanded),
            borderRadius: BorderRadius.circular(6),
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 8, 8, 8),
              child: Row(
                children: [
                  Icon(
                    _iconForTool(widget.call.name),
                    size: 14,
                    color: const Color(0xFFE77E8A),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    widget.call.name,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      fontFamily: 'Cascadia Mono',
                      fontFamilyFallback: ['Consolas', 'monospace'],
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      summary,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        color: Colors.white54,
                        fontSize: 12,
                        fontFamily: 'Cascadia Mono',
                        fontFamilyFallback: ['Consolas', 'monospace'],
                      ),
                    ),
                  ),
                  Icon(
                    _expanded
                        ? Icons.keyboard_arrow_up
                        : Icons.keyboard_arrow_down,
                    size: 14,
                    color: Colors.white54,
                  ),
                ],
              ),
            ),
          ),
          if (_expanded)
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 0, 12, 10),
              child: SelectableText(
                const JsonEncoder.withIndent('  ').convert(args),
                style: const TextStyle(
                  color: Color(0xFFB8BCC8),
                  fontSize: 11,
                  fontFamily: 'Cascadia Mono',
                  fontFamilyFallback: ['Consolas', 'monospace'],
                ),
              ),
            ),
        ],
      ),
    );
  }

  IconData _iconForTool(String name) => switch (name) {
        'read_file' => Icons.description_outlined,
        'write_file' => Icons.edit_note,
        'edit_file' => Icons.edit_outlined,
        'list_files' => Icons.folder_open_outlined,
        'glob_files' => Icons.travel_explore,
        'grep' => Icons.search,
        _ => Icons.build,
      };

  String _summarize(String name, Map<String, dynamic> args) {
    switch (name) {
      case 'read_file':
      case 'list_files':
      case 'write_file':
      case 'edit_file':
        return args['path']?.toString() ?? '';
      case 'glob_files':
        return args['pattern']?.toString() ?? '';
      case 'grep':
        final p = args['pattern'] ?? '';
        final g = args['path_glob'] ?? '**/*';
        return '"$p" in $g';
      default:
        return args.toString();
    }
  }
}

/// Результат tool call — свёрнутая карточка с preview, разворачивается при клике.
class _ToolResultCard extends StatefulWidget {
  const _ToolResultCard({required this.message});
  final Message message;

  @override
  State<_ToolResultCard> createState() => _ToolResultCardState();
}

class _ToolResultCardState extends State<_ToolResultCard> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final content = widget.message.content;
    final firstLine = content.split('\n').first;
    final previewLine = firstLine.length > 80
        ? '${firstLine.substring(0, 80)}…'
        : firstLine;
    final lineCount = '\n'.allMatches(content).length + 1;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Container(
        decoration: BoxDecoration(
          color: const Color(0xFF141517),
          borderRadius: BorderRadius.circular(6),
          border: Border.all(color: Colors.white10),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            InkWell(
              onTap: () => setState(() => _expanded = !_expanded),
              borderRadius: BorderRadius.circular(6),
              child: Padding(
                padding: const EdgeInsets.fromLTRB(12, 6, 8, 6),
                child: Row(
                  children: [
                    const Icon(
                      Icons.output,
                      size: 12,
                      color: Colors.white54,
                    ),
                    const SizedBox(width: 6),
                    const Text(
                      'tool result',
                      style: TextStyle(
                        color: Colors.white54,
                        fontSize: 10,
                        fontFamily: 'Cascadia Mono',
                        fontFamilyFallback: ['Consolas', 'monospace'],
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        previewLine,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          color: Colors.white38,
                          fontSize: 11,
                          fontFamily: 'Cascadia Mono',
                          fontFamilyFallback: ['Consolas', 'monospace'],
                        ),
                      ),
                    ),
                    Text(
                      '$lineCount line${lineCount == 1 ? '' : 's'}',
                      style: const TextStyle(
                        color: Colors.white24,
                        fontSize: 10,
                      ),
                    ),
                    const SizedBox(width: 4),
                    Icon(
                      _expanded
                          ? Icons.keyboard_arrow_up
                          : Icons.keyboard_arrow_down,
                      size: 14,
                      color: Colors.white38,
                    ),
                  ],
                ),
              ),
            ),
            if (_expanded)
              Container(
                padding: const EdgeInsets.fromLTRB(12, 4, 12, 10),
                constraints: const BoxConstraints(maxHeight: 400),
                child: SingleChildScrollView(
                  child: SelectableText(
                    content,
                    style: const TextStyle(
                      color: Color(0xFFB8BCC8),
                      fontSize: 11,
                      height: 1.5,
                      fontFamily: 'Cascadia Mono',
                      fontFamilyFallback: ['Consolas', 'monospace'],
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// Confirmation overlay — показывается поверх всего приложения когда
// ChatController.pendingConfirmations не пуст.
// ════════════════════════════════════════════════════════════════════════
class ConfirmationOverlay extends StatelessWidget {
  const ConfirmationOverlay({super.key});

  @override
  Widget build(BuildContext context) {
    final ctrl = context.watch<ChatController>();
    if (ctrl.pendingConfirmations.isEmpty) return const SizedBox.shrink();

    // Обрабатываем по одному — показываем самый первый.
    final conf = ctrl.pendingConfirmations.first;
    return Positioned.fill(
      child: Container(
        color: const Color(0xCC000000),
        alignment: Alignment.center,
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 720, maxHeight: 600),
          child: Container(
            margin: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: const Color(0xFF1E1F22),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: const Color(0xFFE77E8A)),
              boxShadow: const [
                BoxShadow(color: Colors.black54, blurRadius: 30),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  padding: const EdgeInsets.fromLTRB(20, 16, 20, 12),
                  decoration: const BoxDecoration(
                    border: Border(
                      bottom: BorderSide(color: Colors.white12),
                    ),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.warning_amber,
                          color: Color(0xFFE77E8A), size: 18),
                      const SizedBox(width: 8),
                      Text(
                        'Agent хочет выполнить ${conf.call.name}',
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
                Flexible(
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.all(20),
                    child: SelectableText(
                      conf.diff,
                      style: const TextStyle(
                        color: Color(0xFFE6E7E9),
                        fontSize: 12,
                        height: 1.5,
                        fontFamily: 'Cascadia Mono',
                        fontFamilyFallback: ['Consolas', 'monospace'],
                      ),
                    ),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.fromLTRB(20, 12, 20, 16),
                  decoration: const BoxDecoration(
                    border: Border(top: BorderSide(color: Colors.white12)),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      TextButton(
                        onPressed: () =>
                            ctrl.resolveConfirmation(conf, false),
                        child: const Text('Reject'),
                      ),
                      const SizedBox(width: 8),
                      FilledButton.icon(
                        icon: const Icon(Icons.check, size: 16),
                        label: const Text('Apply'),
                        style: FilledButton.styleFrom(
                          backgroundColor: const Color(0xFFE77E8A),
                        ),
                        onPressed: () =>
                            ctrl.resolveConfirmation(conf, true),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// Settings bottom sheet
// ════════════════════════════════════════════════════════════════════════
class SettingsSheet extends StatefulWidget {
  const SettingsSheet({super.key});

  @override
  State<SettingsSheet> createState() => _SettingsSheetState();
}

String _ctxHintFor(int n) {
  switch (n) {
    case 2048:
      return 'Минимум — быстро, но модель быстро забывает контекст.';
    case 4096:
      return 'Дефолт — сбалансированно. VRAM ~5.7 GB с Qwen 7B Q4.';
    case 8192:
      return 'Удвоенный контекст — впритык на 8 GB VRAM (~7 GB).';
    case 16384:
      return 'На 8 GB VRAM скорее всего будет OOM. Только если есть ≥12 GB.';
    case 32768:
      return 'Максимум Qwen 2.5. Нужно ≥16 GB VRAM. На 8 GB не запустится.';
    default:
      return '';
  }
}

class _SettingsSheetState extends State<SettingsSheet> {
  final _systemPromptController = TextEditingController();
  final _baseUrlController = TextEditingController();
  bool _initialised = false;

  @override
  void dispose() {
    _systemPromptController.dispose();
    _baseUrlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ctrl = context.watch<ChatController>();
    final conv = ctrl.activeConversation;

    if (!_initialised) {
      if (conv != null) {
        _systemPromptController.text = conv.systemPrompt;
      }
      _baseUrlController.text = ctrl.ollamaBaseUrl;
      _initialised = true;
    }

    return Padding(
      padding:
          EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
      child: Container(
        padding: const EdgeInsets.all(24),
        constraints: const BoxConstraints(maxHeight: 600),
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Chat settings',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 20),

              // ── Preset selector ──
              const Text(
                'System prompt preset',
                style: TextStyle(color: Colors.white70, fontSize: 13),
              ),
              const SizedBox(height: 6),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: kSystemPresets.entries
                    .map(
                      (e) => ActionChip(
                        label: Text(
                          e.key,
                          style: const TextStyle(fontSize: 11),
                        ),
                        backgroundColor: const Color(0xFF26272A),
                        side: const BorderSide(color: Colors.white12),
                        onPressed: () {
                          _systemPromptController.text = e.value;
                          ctrl.updateActiveSystemPrompt(e.value);
                        },
                      ),
                    )
                    .toList(),
              ),
              const SizedBox(height: 12),

              // ── System prompt editor ──
              TextField(
                controller: _systemPromptController,
                maxLines: 8,
                minLines: 4,
                style: const TextStyle(color: Colors.white, fontSize: 12),
                decoration: const InputDecoration(
                  hintText: 'System prompt…',
                ),
                onChanged: (v) => ctrl.updateActiveSystemPrompt(v),
              ),
              const SizedBox(height: 20),

              // ── Temperature ──
              Row(
                children: [
                  const Text(
                    'Temperature',
                    style: TextStyle(color: Colors.white70, fontSize: 13),
                  ),
                  const Spacer(),
                  Text(
                    (conv?.temperature ?? 0.3).toStringAsFixed(2),
                    style: const TextStyle(color: Colors.white, fontSize: 13),
                  ),
                ],
              ),
              Slider(
                value: conv?.temperature ?? 0.3,
                min: 0,
                max: 2,
                divisions: 40,
                onChanged: conv == null
                    ? null
                    : (v) => ctrl.updateActiveTemperature(v),
              ),
              const SizedBox(height: 12),

              // ── RAG ──
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text(
                  'Use codebase context (Qdrant RAG)',
                  style: TextStyle(color: Colors.white, fontSize: 13),
                ),
                subtitle: const Text(
                  'Перед отправкой в LLM подмешивает top-5 релевантных кусков кода.',
                  style: TextStyle(color: Colors.white54, fontSize: 11),
                ),
                value: ctrl.useCodebaseContext,
                onChanged: (v) => ctrl.useCodebaseContext = v,
              ),
              const SizedBox(height: 12),

              // ── Context window ──
              Row(
                children: [
                  const Text(
                    'Context window (numCtx)',
                    style: TextStyle(color: Colors.white70, fontSize: 13),
                  ),
                  const Spacer(),
                  Text(
                    '${ctrl.contextWindow}',
                    style: const TextStyle(color: Colors.white, fontSize: 13),
                  ),
                ],
              ),
              const SizedBox(height: 6),
              Wrap(
                spacing: 6,
                children: const [2048, 4096, 8192, 16384, 32768]
                    .map((n) => ChoiceChip(
                          label: Text(
                            n >= 1024 ? '${n ~/ 1024}K' : '$n',
                            style: const TextStyle(fontSize: 11),
                          ),
                          selected: ctrl.contextWindow == n,
                          backgroundColor: const Color(0xFF26272A),
                          selectedColor: const Color(0xFF7C9CFF),
                          onSelected: (v) {
                            if (v) ctrl.contextWindow = n;
                          },
                        ))
                    .toList(),
              ),
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  _ctxHintFor(ctrl.contextWindow),
                  style: TextStyle(
                    color: ctrl.contextWindow > 8192
                        ? Colors.orange.shade300
                        : Colors.white38,
                    fontSize: 10,
                  ),
                ),
              ),
              const SizedBox(height: 20),

              // ── Workspace (для agent mode) ──
              const Text(
                'Workspace для agent mode',
                style: TextStyle(color: Colors.white70, fontSize: 13),
              ),
              const SizedBox(height: 6),
              Row(
                children: [
                  Expanded(
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 10),
                      decoration: BoxDecoration(
                        color: const Color(0xFF26272A),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        ctrl.workspaceRoot.isEmpty
                            ? '(не выбран — agent mode недоступен)'
                            : ctrl.workspaceRoot,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          color: ctrl.workspaceRoot.isEmpty
                              ? Colors.white38
                              : Colors.white,
                          fontSize: 12,
                          fontFamily: 'Cascadia Mono',
                          fontFamilyFallback: const ['Consolas', 'monospace'],
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  FilledButton(
                    onPressed: () async {
                      final selected =
                          await FilePicker.platform.getDirectoryPath(
                        dialogTitle: 'Select workspace for agent mode',
                      );
                      if (selected != null) {
                        await ctrl.setWorkspaceRoot(selected);
                        if (context.mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: Text('Workspace: $selected'),
                              duration: const Duration(seconds: 2),
                              behavior: SnackBarBehavior.floating,
                            ),
                          );
                        }
                      }
                    },
                    child: const Text('Browse…'),
                  ),
                ],
              ),
              const Padding(
                padding: EdgeInsets.only(top: 6),
                child: Text(
                  'Agent сможет читать и править файлы только внутри этой папки.',
                  style: TextStyle(color: Colors.white38, fontSize: 10),
                ),
              ),
              const SizedBox(height: 20),

              // ── Ollama URL ──
              const Text(
                'Ollama base URL',
                style: TextStyle(color: Colors.white70, fontSize: 13),
              ),
              const SizedBox(height: 6),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _baseUrlController,
                      style: const TextStyle(color: Colors.white, fontSize: 12),
                      decoration: const InputDecoration(
                        hintText: 'http://localhost:11434',
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  FilledButton(
                    onPressed: () {
                      ctrl.setOllamaBaseUrl(_baseUrlController.text);
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text('Ollama URL updated'),
                          duration: Duration(seconds: 1),
                          behavior: SnackBarBehavior.floating,
                        ),
                      );
                    },
                    child: const Text('Apply'),
                  ),
                ],
              ),
              const SizedBox(height: 16),

              Align(
                alignment: Alignment.centerRight,
                child: TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('Close'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
