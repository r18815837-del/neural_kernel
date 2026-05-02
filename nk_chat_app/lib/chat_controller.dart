import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:uuid/uuid.dart';

import 'models.dart';
import 'ollama_service.dart';
import 'persistence.dart';
import 'rag_service.dart';
import 'tools.dart';

const _uuid = Uuid();

/// Запрос на подтверждение tool call — UI смотрит на
/// controller.pendingConfirmations и показывает диалог.
class PendingToolConfirmation {
  PendingToolConfirmation({required this.call, required this.diff});
  final ToolCall call;
  final String diff; // готовый текст для отображения
  final Completer<bool> completer = Completer<bool>();
}

/// Готовые system-prompt пресеты — переключаются одним кликом в настройках.
const kSystemPresets = <String, String>{
  'Default (neural_kernel)': '''
You are a programming assistant running locally via Ollama.
The user's main project is `neural_kernel` — a from-scratch deep learning
framework (NumPy + optional CuPy) with autograd, Transformers, BPE tokenization,
and a language-model training pipeline.

Be direct and concise. Show code examples, explain only when asked. Prefer
Python 3.10+ idioms. Avoid disclaimers.
''',
  'Code reviewer': '''
You are a senior code reviewer. When given code:
1. Point out real bugs and logical errors first.
2. Then call out style issues (naming, structure, complexity).
3. Finally suggest improvements or alternative approaches.
Be direct, no hedging. Output a bullet list of findings, not a long essay.
''',
  'Refactor specialist': '''
You refactor code to be cleaner, shorter, more idiomatic. Keep behavior
identical. Add brief comments only where intent isn't obvious. Output the
refactored code first, then a short "What changed" summary.
''',
  'Debug helper': '''
You are helping debug a problem. Ask for relevant info first (stack traces,
inputs, expected vs actual). Form hypotheses, suggest minimum reproduction
cases. When you propose a fix, explain WHY the bug happened.
''',
  'ELI5 teacher': '''
You explain complex programming topics in simple terms, using analogies from
everyday life. Avoid jargon unless defined. Short sentences. Your student
knows basic Python but nothing advanced.
''',
  'Plain chat': '''
You are a helpful assistant. Answer concisely.
''',
};

/// Центральный state-holder приложения.
class ChatController extends ChangeNotifier {
  ChatController({
    OllamaService? service,
    RagService? ragService,
  })  : _service = service ?? OllamaService(),
        _rag = ragService ?? RagService();

  OllamaService _service;
  final RagService _rag;

  // ── models ──
  List<String> _availableModels = [];
  String _selectedModel = 'qwen2.5-coder:7b';
  String _ollamaBaseUrl = 'http://localhost:11434';

  List<String> get availableModels => List.unmodifiable(_availableModels);
  String get selectedModel => _selectedModel;
  String get ollamaBaseUrl => _ollamaBaseUrl;

  set selectedModel(String m) {
    _selectedModel = m;
    _persistSettings();
    notifyListeners();
  }

  Future<void> setOllamaBaseUrl(String url) async {
    _ollamaBaseUrl = url.trim();
    _service.dispose();
    _service = OllamaService(baseUrl: _ollamaBaseUrl);
    await refreshModels();
    _persistSettings();
  }

  // ── conversations ──
  final List<Conversation> _conversations = [];
  String? _activeId;
  String _searchQuery = '';

  String get searchQuery => _searchQuery;

  set searchQuery(String q) {
    _searchQuery = q;
    notifyListeners();
  }

  List<Conversation> get conversations {
    final list = List<Conversation>.from(_conversations);
    list.sort((a, b) => b.updatedAt.compareTo(a.updatedAt));
    if (_searchQuery.trim().isEmpty) return List.unmodifiable(list);
    final q = _searchQuery.toLowerCase();
    return List.unmodifiable(
      list.where((c) {
        if (c.title.toLowerCase().contains(q)) return true;
        return c.messages.any((m) => m.content.toLowerCase().contains(q));
      }),
    );
  }

  Conversation? get activeConversation {
    if (_activeId == null) return null;
    try {
      return _conversations.firstWhere((c) => c.id == _activeId);
    } catch (_) {
      return null;
    }
  }

  // ── RAG toggle ──
  bool _useCodebaseContext = false;

  bool get useCodebaseContext => _useCodebaseContext;

  set useCodebaseContext(bool v) {
    _useCodebaseContext = v;
    _persistSettings();
    notifyListeners();
  }

  // ── Agent mode + workspace ──
  bool _agentMode = false;
  String _workspaceRoot = '';

  bool get agentMode => _agentMode;
  String get workspaceRoot => _workspaceRoot;

  // ── Code panel state (split layout) ──
  String? _panelFilePath;
  String? _panelFileContent;
  ToolCall? _panelPreviewEdit;

  String? get panelFilePath => _panelFilePath;
  String? get panelFileContent => _panelFileContent;
  ToolCall? get panelPreviewEdit => _panelPreviewEdit;

  void openFileInPanel(String path, String content) {
    _panelFilePath = path;
    _panelFileContent = content;
    _panelPreviewEdit = null;
    notifyListeners();
  }

  void showEditPreview(ToolCall call) {
    _panelPreviewEdit = call;
    // Если редактируемый файл уже открыт — оставляем content для контекста.
    final path = call.arguments['path'] as String?;
    if (path != null && _panelFilePath != path) {
      // Попробуем прочитать заново, чтобы diff был осмысленным.
      try {
        final abs = _workspaceRoot.isEmpty
            ? null
            : '$_workspaceRoot${Platform.pathSeparator}$path';
        if (abs != null) {
          final f = File(abs);
          if (f.existsSync()) {
            _panelFilePath = path;
            _panelFileContent = f.readAsStringSync();
          }
        }
      } catch (_) {}
    }
    notifyListeners();
  }

  void clearPanel() {
    _panelFilePath = null;
    _panelFileContent = null;
    _panelPreviewEdit = null;
    notifyListeners();
  }

  // ── Context window (numCtx) ──
  int _contextWindow = 4096;
  int get contextWindow => _contextWindow;

  set contextWindow(int n) {
    _contextWindow = n;
    _persistSettings();
    notifyListeners();
  }

  set agentMode(bool v) {
    _agentMode = v;
    _persistSettings();
    notifyListeners();
  }

  Future<void> setWorkspaceRoot(String path) async {
    _workspaceRoot = path.trim();
    await _persistSettings();
    notifyListeners();
  }

  // ── Pending tool confirmations ──
  final List<PendingToolConfirmation> _pendingConfirmations = [];

  List<PendingToolConfirmation> get pendingConfirmations =>
      List.unmodifiable(_pendingConfirmations);

  void resolveConfirmation(PendingToolConfirmation conf, bool approved) {
    if (!_pendingConfirmations.remove(conf)) return;
    if (!conf.completer.isCompleted) conf.completer.complete(approved);
    notifyListeners();
  }

  // ── flags ──
  bool _streaming = false;
  bool _ragSearching = false;
  String? _lastError;

  bool get streaming => _streaming;
  bool get ragSearching => _ragSearching;
  String? get lastError => _lastError;

  void clearError() {
    _lastError = null;
    notifyListeners();
  }

  // ══════════════════════════════════════════════════════════════════════
  // Init — персистентность + загрузка моделей.
  // ══════════════════════════════════════════════════════════════════════
  Future<void> init() async {
    // Настройки.
    final s = await Persistence.loadSettings();
    if (s['ollamaBaseUrl'] is String) {
      _ollamaBaseUrl = s['ollamaBaseUrl'] as String;
      _service.dispose();
      _service = OllamaService(baseUrl: _ollamaBaseUrl);
    }
    if (s['selectedModel'] is String) {
      _selectedModel = s['selectedModel'] as String;
    }
    if (s['useCodebaseContext'] is bool) {
      _useCodebaseContext = s['useCodebaseContext'] as bool;
    }
    if (s['agentMode'] is bool) {
      _agentMode = s['agentMode'] as bool;
    }
    if (s['workspaceRoot'] is String) {
      _workspaceRoot = s['workspaceRoot'] as String;
    }
    if (s['contextWindow'] is int) {
      _contextWindow = s['contextWindow'] as int;
    }

    // Чаты.
    final loaded = await Persistence.loadConversations();
    if (loaded.isNotEmpty) {
      _conversations.addAll(loaded);
      _activeId = loaded.first.id;
    }

    // Модели из Ollama.
    await refreshModels();

    notifyListeners();
  }

  Future<void> _persistSettings() async {
    await Persistence.saveSettings({
      'ollamaBaseUrl': _ollamaBaseUrl,
      'selectedModel': _selectedModel,
      'useCodebaseContext': _useCodebaseContext,
      'agentMode': _agentMode,
      'workspaceRoot': _workspaceRoot,
      'contextWindow': _contextWindow,
    });
  }

  void _schedulePersistChats() {
    Persistence.scheduleSave(_conversations);
  }

  // ══════════════════════════════════════════════════════════════════════
  // Models
  // ══════════════════════════════════════════════════════════════════════
  static bool _isEmbeddingModel(String name) {
    final n = name.toLowerCase();
    return n.contains('embed') ||
        n.contains('-e5-') ||
        n.startsWith('bge-') ||
        n.contains('/bge-') ||
        n.startsWith('all-minilm') ||
        n.contains('snowflake-arctic-embed');
  }

  Future<void> refreshModels() async {
    try {
      final all = await _service.listModels();
      _availableModels = all.where((m) => !_isEmbeddingModel(m)).toList();

      if (_availableModels.isNotEmpty &&
          !_availableModels.contains(_selectedModel)) {
        _selectedModel = _availableModels.first;
      }

      final conv = activeConversation;
      if (conv != null &&
          _availableModels.isNotEmpty &&
          (!_availableModels.contains(conv.model) ||
              _isEmbeddingModel(conv.model))) {
        conv.model = _availableModels.first;
      }

      _lastError = null;
    } catch (e) {
      _lastError = e.toString();
    }
    notifyListeners();
  }

  // ══════════════════════════════════════════════════════════════════════
  // Conversation CRUD
  // ══════════════════════════════════════════════════════════════════════
  Conversation createConversation({String? model}) {
    final c = Conversation(
      model: model ?? _selectedModel,
      systemPrompt: kSystemPresets['Default (neural_kernel)']!,
    );
    _conversations.add(c);
    _activeId = c.id;
    _schedulePersistChats();
    notifyListeners();
    return c;
  }

  void selectConversation(String id) {
    _activeId = id;
    notifyListeners();
  }

  void deleteConversation(String id) {
    _conversations.removeWhere((c) => c.id == id);
    if (_activeId == id) {
      _activeId = _conversations.isEmpty ? null : _conversations.last.id;
    }
    _schedulePersistChats();
    notifyListeners();
  }

  void renameConversation(String id, String newTitle) {
    final c = _conversations.firstWhere(
      (c) => c.id == id,
      orElse: () => throw ArgumentError('conversation not found'),
    );
    c.title = newTitle.trim().isEmpty ? 'Untitled' : newTitle.trim();
    c.touch();
    _schedulePersistChats();
    notifyListeners();
  }

  void updateActiveSystemPrompt(String text) {
    final c = activeConversation;
    if (c == null) return;
    c.systemPrompt = text;
    c.touch();
    _schedulePersistChats();
    notifyListeners();
  }

  void updateActiveTemperature(double t) {
    final c = activeConversation;
    if (c == null) return;
    c.temperature = t;
    c.touch();
    _schedulePersistChats();
    notifyListeners();
  }

  void updateActiveModel(String model) {
    final c = activeConversation;
    if (c == null) return;
    c.model = model;
    c.touch();
    _schedulePersistChats();
    notifyListeners();
  }

  // ══════════════════════════════════════════════════════════════════════
  // Messaging
  // ══════════════════════════════════════════════════════════════════════
  Future<void> sendUserMessage(String text) async {
    final trimmed = text.trim();
    if (trimmed.isEmpty) return;
    if (_streaming) return;

    var conv = activeConversation;
    conv ??= createConversation();

    conv.messages.add(Message(role: MessageRole.user, content: trimmed));
    conv.touch();

    if (conv.title == 'New chat') {
      conv.autoRename();
    }

    // Agent mode — отдельный путь с tool calls.
    if (_agentMode && _workspaceRoot.isNotEmpty) {
      await _runAgentLoop(conv);
      return;
    }

    final assistantMsg = Message(
      role: MessageRole.assistant,
      content: '',
      streaming: true,
    );
    conv.messages.add(assistantMsg);
    await _runStream(conv, assistantMsg, trimmed);
  }

  /// Основной стриминговый «раннер». Используется и sendUserMessage,
  /// и regenerateLastAssistant, и editUserMessage.
  Future<void> _runStream(
    Conversation conv,
    Message assistantMsg,
    String lastUserText,
  ) async {
    _streaming = true;
    _lastError = null;
    notifyListeners();

    // Если включён RAG — пытаемся подмешать контекст из кодовой базы.
    String systemPromptOverride = conv.systemPrompt;
    List<CodeChunk> retrievedChunks = const [];
    if (_useCodebaseContext) {
      _ragSearching = true;
      notifyListeners();
      try {
        retrievedChunks = await _rag.search(lastUserText, limit: 5);
        final block = _rag.buildContextBlock(retrievedChunks);
        if (block != null) {
          systemPromptOverride =
              '${conv.systemPrompt.trim()}\n\n$block';
        }
      } catch (e) {
        debugPrint('RAG failed: $e');
      } finally {
        _ragSearching = false;
        notifyListeners();
      }
    }

    // Привязываем use chunks к ответу для показа под сообщением.
    if (retrievedChunks.isNotEmpty) {
      assistantMsg.ragRefs = retrievedChunks
          .map((c) => MessageRagRef(
                file: c.file,
                startLine: c.startLine,
                endLine: c.endLine,
                kind: c.kind,
                name: c.name,
                score: c.score,
              ))
          .toList();
    }

    final messagesForApi = <Map<String, dynamic>>[];
    if (systemPromptOverride.trim().isNotEmpty) {
      messagesForApi.add({'role': 'system', 'content': systemPromptOverride.trim()});
    }
    for (final m in conv.messages) {
      if (m.streaming && m.content.isEmpty) continue;
      messagesForApi.add({'role': m.role.name, 'content': m.content});
    }

    try {
      await _service.chatStream(
        model: conv.model,
        messages: messagesForApi,
        temperature: conv.temperature,
        numCtx: _contextWindow,
        onToken: (token) {
          assistantMsg.content += token;
          notifyListeners();
        },
        onDone: () {
          assistantMsg.streaming = false;
          _streaming = false;
          conv.touch();
          _schedulePersistChats();
          notifyListeners();
        },
        onError: (e) {
          _lastError = e.toString();
          if (assistantMsg.content.isEmpty) {
            assistantMsg.content = '_[ошибка: $e]_';
          }
          assistantMsg.streaming = false;
          _streaming = false;
          _schedulePersistChats();
          notifyListeners();
        },
      );
    } catch (e) {
      _lastError = e.toString();
      assistantMsg.streaming = false;
      _streaming = false;
      _schedulePersistChats();
      notifyListeners();
    }
  }

  Future<void> cancelGeneration() async {
    if (!_streaming) return;
    await _service.cancel();
    final conv = activeConversation;
    if (conv != null) {
      for (final m in conv.messages) {
        if (m.streaming) m.streaming = false;
      }
    }
    _streaming = false;
    _schedulePersistChats();
    notifyListeners();
  }

  Future<void> regenerateLastAssistant() async {
    final conv = activeConversation;
    if (conv == null || _streaming) return;
    if (conv.messages.isEmpty) return;
    if (conv.messages.last.role != MessageRole.assistant) return;

    conv.messages.removeLast();
    // найти последний user-текст
    final lastUser = conv.messages.lastWhere(
      (m) => m.role == MessageRole.user,
      orElse: () => Message(role: MessageRole.user, content: ''),
    );

    final newAssistant = Message(
      role: MessageRole.assistant,
      content: '',
      streaming: true,
    );
    conv.messages.add(newAssistant);
    conv.touch();

    await _runStream(conv, newAssistant, lastUser.content);
  }

  /// Редактирование user-сообщения: переписываем его текст, удаляем всё
  /// что было после, перезапрашиваем assistant.
  Future<void> editUserMessage(String messageId, String newText) async {
    final conv = activeConversation;
    if (conv == null || _streaming) return;

    final idx = conv.messages.indexWhere((m) => m.id == messageId);
    if (idx < 0) return;
    final msg = conv.messages[idx];
    if (msg.role != MessageRole.user) return;

    msg.content = newText.trim();
    // удаляем всё после.
    if (idx + 1 < conv.messages.length) {
      conv.messages.removeRange(idx + 1, conv.messages.length);
    }

    final newAssistant = Message(
      role: MessageRole.assistant,
      content: '',
      streaming: true,
    );
    conv.messages.add(newAssistant);
    conv.touch();

    await _runStream(conv, newAssistant, msg.content);
  }

  void deleteMessage(String messageId) {
    final conv = activeConversation;
    if (conv == null) return;
    conv.messages.removeWhere((m) => m.id == messageId);
    conv.touch();
    _schedulePersistChats();
    notifyListeners();
  }

  /// Отправить предопределённый follow-up к последнему сообщению.
  /// Используется quick-action кнопками (Shorter / Simpler / Translate).
  Future<void> sendFollowUp(String prompt) async {
    if (_streaming) return;
    await sendUserMessage(prompt);
  }

  // ══════════════════════════════════════════════════════════════════════
  // Agent loop: многоходовое выполнение с tool calls
  // ══════════════════════════════════════════════════════════════════════
  static const int _agentMaxTurns = 10;

  Future<void> _runAgentLoop(Conversation conv) async {
    _streaming = true;
    _lastError = null;
    notifyListeners();

    final executor = ToolExecutor(
      workspaceRoot: _workspaceRoot,
      confirm: (call) async {
        final diff = _buildDiffForConfirmation(call);
        // Показываем preview правке в code-панели.
        showEditPreview(call);
        final conf = PendingToolConfirmation(call: call, diff: diff);
        _pendingConfirmations.add(conf);
        notifyListeners();
        return conf.completer.future;
      },
    );

    // Системный промпт для agent mode: даём понять что есть tools.
    final agentSystemPrompt = _agentSystemPrompt(conv);

    try {
      for (var turn = 0; turn < _agentMaxTurns; turn++) {
        // Собираем сообщения для Ollama.
        final messagesForApi = <Map<String, dynamic>>[
          {'role': 'system', 'content': agentSystemPrompt},
          ...conv
              .toOllamaMessages()
              .where((m) => m['role'] != 'system'), // сис-промпт уже добавили
        ];

        // Вставляем пустое assistant-сообщение для typing-dots.
        final placeholder = Message(
          role: MessageRole.assistant,
          content: '',
          streaming: true,
        );
        conv.messages.add(placeholder);
        notifyListeners();

        ChatTurn turnResult;
        try {
          turnResult = await _service.chat(
            model: conv.model,
            messages: messagesForApi,
            tools: kToolSpecs.map((t) => t.toOllamaJson()).toList(),
            temperature: conv.temperature,
            numCtx: _contextWindow,
          );
        } catch (e) {
          // Убираем плейсхолдер, показываем ошибку.
          conv.messages.remove(placeholder);
          placeholder.streaming = false;
          placeholder.content = '_[ошибка: $e]_';
          conv.messages.add(placeholder);
          _lastError = e.toString();
          break;
        }

        // Заменяем плейсхолдер настоящим сообщением.
        conv.messages.remove(placeholder);
        final assistantMsg = Message(
          role: MessageRole.assistant,
          content: turnResult.content,
          toolCalls: turnResult.toolCalls.isEmpty
              ? null
              : turnResult.toolCalls,
        );
        conv.messages.add(assistantMsg);
        conv.touch();
        notifyListeners();

        // Если нет tool calls — финальный ответ, выходим.
        if (turnResult.toolCalls.isEmpty) break;

        // Выполняем каждый tool call, добавляем результат.
        for (final call in turnResult.toolCalls) {
          final result = await executor.execute(call);

          // Side-effect: отобразить результаты read/edit в code-панели.
          if (result.success) {
            if (call.name == 'read_file') {
              final path = call.arguments['path'] as String? ?? '';
              if (path.isNotEmpty) openFileInPanel(path, result.content);
            } else if (call.name == 'write_file' ||
                call.name == 'edit_file') {
              // После успешной правки — перечитываем файл для актуального content.
              final path = call.arguments['path'] as String? ?? '';
              if (path.isNotEmpty) {
                try {
                  final abs =
                      '$_workspaceRoot${Platform.pathSeparator}$path';
                  final f = File(abs);
                  if (f.existsSync()) {
                    openFileInPanel(path, f.readAsStringSync());
                  }
                } catch (_) {}
              }
            }
          }

          conv.messages.add(Message(
            role: MessageRole.tool,
            content: result.content,
            toolCallId: call.id,
          ));
          conv.touch();
          notifyListeners();

          // Если пользователь отклонил write — прерываем всю цепочку,
          // не гоняем модель зря.
          if (result.rejected) {
            conv.messages.add(Message(
              role: MessageRole.assistant,
              content: '_(agent остановлен пользователем)_',
            ));
            _schedulePersistChats();
            _streaming = false;
            notifyListeners();
            return;
          }
        }
        _schedulePersistChats();
      }

      // Дошли до max turns.
    } finally {
      _streaming = false;
      _schedulePersistChats();
      notifyListeners();
    }
  }

  String _agentSystemPrompt(Conversation conv) {
    final baseSystem = conv.systemPrompt.trim();
    final toolsHint = '''
You have access to the user's workspace via tools. The workspace root is:
  $_workspaceRoot

Use the tools to help with concrete file questions:
  - list_files(path) — inspect the directory structure
  - read_file(path) — read a file's content
  - glob_files(pattern) — find files by pattern
  - grep(pattern, path_glob) — search text in files
  - write_file(path, content) — create/overwrite (needs user approval)
  - edit_file(path, old_string, new_string) — replace a unique string (needs approval)

Guidelines:
  - Prefer read/grep/glob to understand before writing.
  - Use edit_file for small targeted changes. Provide enough surrounding
    context in old_string so it matches exactly one spot.
  - If a tool fails, read the error and try a different approach.
  - When done, give a final text answer to the user.
''';
    return baseSystem.isEmpty ? toolsHint : '$baseSystem\n\n$toolsHint';
  }

  /// Готовит человеко-читаемый diff-текст для confirmation dialog.
  String _buildDiffForConfirmation(ToolCall call) {
    if (call.name == 'write_file') {
      final path = call.arguments['path'] as String? ?? '?';
      final content = call.arguments['content'] as String? ?? '';
      final preview = content.length > 2000
          ? '${content.substring(0, 2000)}\n... (${content.length} chars total)'
          : content;
      return 'Write file: $path\n\n--- NEW CONTENT ---\n$preview';
    }
    if (call.name == 'edit_file') {
      final path = call.arguments['path'] as String? ?? '?';
      final oldStr = call.arguments['old_string'] as String? ?? '';
      final newStr = call.arguments['new_string'] as String? ?? '';
      return 'Edit file: $path\n\n'
          '--- REMOVE ---\n$oldStr\n\n'
          '--- REPLACE WITH ---\n$newStr';
    }
    return 'Tool: ${call.name}\nArguments: ${call.arguments}';
  }

  // ══════════════════════════════════════════════════════════════════════
  // Export
  // ══════════════════════════════════════════════════════════════════════
  /// Собирает активный чат как Markdown-строку.
  String exportActiveAsMarkdown() {
    final conv = activeConversation;
    if (conv == null) return '';
    final buf = StringBuffer();
    buf.writeln('# ${conv.title}');
    buf.writeln();
    buf.writeln('_Model: ${conv.model} · temperature: ${conv.temperature} · '
        'created ${conv.createdAt.toLocal()}_');
    buf.writeln();
    for (final m in conv.messages) {
      if (m.role == MessageRole.system) continue;
      final who = m.role == MessageRole.user ? '## You' : '## Assistant';
      buf.writeln(who);
      buf.writeln();
      buf.writeln(m.content.trim());
      buf.writeln();
    }
    return buf.toString();
  }

  /// Сохранить экспорт в файл. Возвращает путь.
  Future<String> exportActiveToFile(String dirPath) async {
    final conv = activeConversation;
    if (conv == null) throw StateError('no active conversation');

    final safeName = conv.title
        .replaceAll(RegExp(r'[^\w\s-]'), '')
        .replaceAll(RegExp(r'\s+'), '_');
    final timestamp = DateTime.now()
        .toIso8601String()
        .split('.')
        .first
        .replaceAll(':', '-');
    final path = '$dirPath${Platform.pathSeparator}${safeName}_$timestamp.md';
    await File(path).writeAsString(exportActiveAsMarkdown(), flush: true);
    return path;
  }

  // ══════════════════════════════════════════════════════════════════════
  // Shutdown
  // ══════════════════════════════════════════════════════════════════════
  Future<void> flushToDisk() async {
    await Persistence.flush(_conversations);
  }

  @override
  void dispose() {
    _service.dispose();
    super.dispose();
  }
}
