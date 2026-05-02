import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import 'models.dart';
import 'tools.dart';

/// Сериализация и хранение состояния NK Chat на диске.
///
/// Данные лежат в %APPDATA%/nk_chat/ (Windows), `~/.config/nk_chat/` (Linux),
/// `~/Library/Application Support/nk_chat/` (macOS).
///
/// Файлы:
///   conversations.json — все чаты
///   settings.json      — пользовательские настройки
class Persistence {
  Persistence._();

  static Directory? _dir;

  static Future<Directory> _ensureDir() async {
    if (_dir != null) return _dir!;
    final base = await getApplicationSupportDirectory();
    final target = Directory('${base.path}${Platform.pathSeparator}nk_chat');
    if (!await target.exists()) {
      await target.create(recursive: true);
    }
    _dir = target;
    return target;
  }

  // ── Conversations ──────────────────────────────────────────────────────
  static Future<List<Conversation>> loadConversations() async {
    try {
      final dir = await _ensureDir();
      final file = File('${dir.path}${Platform.pathSeparator}conversations.json');
      if (!await file.exists()) return [];
      final data = jsonDecode(await file.readAsString()) as List;
      return data
          .map((e) => _ConversationIO.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (_) {
      return [];
    }
  }

  static Future<void> saveConversations(List<Conversation> conversations) async {
    try {
      final dir = await _ensureDir();
      final file = File('${dir.path}${Platform.pathSeparator}conversations.json');
      final json = conversations
          .map((c) => _ConversationIO.toJson(c))
          .toList();
      await file.writeAsString(jsonEncode(json), flush: true);
    } catch (_) {
      // Тихо проглатываем — не хочется падать, если диск полный.
    }
  }

  // ── Settings ───────────────────────────────────────────────────────────
  static Future<Map<String, dynamic>> loadSettings() async {
    try {
      final dir = await _ensureDir();
      final file = File('${dir.path}${Platform.pathSeparator}settings.json');
      if (!await file.exists()) return {};
      return jsonDecode(await file.readAsString()) as Map<String, dynamic>;
    } catch (_) {
      return {};
    }
  }

  static Future<void> saveSettings(Map<String, dynamic> settings) async {
    try {
      final dir = await _ensureDir();
      final file = File('${dir.path}${Platform.pathSeparator}settings.json');
      await file.writeAsString(jsonEncode(settings), flush: true);
    } catch (_) {}
  }

  /// Debouncer — задерживает save на `delay`, чтобы не писать диск на каждый токен.
  static Timer? _saveTimer;

  static void scheduleSave(
    List<Conversation> conversations, {
    Duration delay = const Duration(milliseconds: 800),
  }) {
    _saveTimer?.cancel();
    _saveTimer = Timer(delay, () => saveConversations(conversations));
  }

  /// Принудительно сохранить сейчас, если есть pending save.
  static Future<void> flush(List<Conversation> conversations) async {
    _saveTimer?.cancel();
    _saveTimer = null;
    await saveConversations(conversations);
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────
extension _ConversationIO on Conversation {
  static Map<String, dynamic> toJson(Conversation c) => {
        'id': c.id,
        'title': c.title,
        'model': c.model,
        'systemPrompt': c.systemPrompt,
        'temperature': c.temperature,
        'messages': c.messages.map(_messageToJson).toList(),
        'createdAt': c.createdAt.toIso8601String(),
        'updatedAt': c.updatedAt.toIso8601String(),
      };

  static Conversation fromJson(Map<String, dynamic> json) {
    final c = Conversation(
      id: json['id'] as String?,
      title: json['title'] as String? ?? 'New chat',
      model: json['model'] as String? ?? 'qwen2.5-coder:7b',
      systemPrompt: json['systemPrompt'] as String? ?? '',
      temperature: (json['temperature'] as num?)?.toDouble() ?? 0.3,
      createdAt: DateTime.tryParse(json['createdAt'] as String? ?? ''),
      updatedAt: DateTime.tryParse(json['updatedAt'] as String? ?? ''),
    );
    final msgs = (json['messages'] as List? ?? [])
        .map((e) => _messageFromJson(e as Map<String, dynamic>))
        .toList();
    c.messages.addAll(msgs);
    return c;
  }
}

Map<String, dynamic> _messageToJson(Message m) => {
      'id': m.id,
      'role': m.role.name,
      'content': m.content,
      'createdAt': m.createdAt.toIso8601String(),
      if (m.ragRefs != null)
        'ragRefs': m.ragRefs!.map((r) => r.toJson()).toList(),
      if (m.toolCalls != null)
        'toolCalls': m.toolCalls!.map((c) => c.toJson()).toList(),
      if (m.toolCallId != null) 'toolCallId': m.toolCallId,
    };

Message _messageFromJson(Map<String, dynamic> json) {
  final roleStr = json['role'] as String;
  final role = switch (roleStr) {
    'system' => MessageRole.system,
    'assistant' => MessageRole.assistant,
    'tool' => MessageRole.tool,
    _ => MessageRole.user,
  };
  final refs = (json['ragRefs'] as List?)
      ?.map((e) => MessageRagRef.fromJson(e as Map<String, dynamic>))
      .toList();
  final calls = (json['toolCalls'] as List?)
      ?.map<ToolCall>((e) => ToolCall.fromJson(e as Map<String, dynamic>))
      .toList();
  return Message(
    id: json['id'] as String?,
    role: role,
    content: json['content'] as String? ?? '',
    createdAt: DateTime.tryParse(json['createdAt'] as String? ?? ''),
    ragRefs: refs,
    toolCalls: calls,
    toolCallId: json['toolCallId'] as String?,
  );
}
