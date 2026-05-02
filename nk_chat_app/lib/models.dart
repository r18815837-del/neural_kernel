import 'package:uuid/uuid.dart';

import 'tools.dart';

const _uuid = Uuid();

enum MessageRole { system, user, assistant, tool }

MessageRole _roleFromString(String s) => switch (s) {
      'system' => MessageRole.system,
      'user' => MessageRole.user,
      'assistant' => MessageRole.assistant,
      'tool' => MessageRole.tool,
      _ => MessageRole.user,
    };

String _roleToString(MessageRole r) => r.name;

/// Краткий референс на чанк кода, который использовался как контекст RAG.
/// Храним в Message, чтобы показать пользователю под ответом.
class MessageRagRef {
  const MessageRagRef({
    required this.file,
    required this.startLine,
    required this.endLine,
    required this.kind,
    required this.name,
    required this.score,
  });

  final String file;
  final int startLine;
  final int endLine;
  final String kind;
  final String name;
  final double score;

  Map<String, dynamic> toJson() => {
        'file': file,
        'startLine': startLine,
        'endLine': endLine,
        'kind': kind,
        'name': name,
        'score': score,
      };

  factory MessageRagRef.fromJson(Map<String, dynamic> json) => MessageRagRef(
        file: json['file'] as String,
        startLine: (json['startLine'] as num).toInt(),
        endLine: (json['endLine'] as num).toInt(),
        kind: json['kind'] as String,
        name: json['name'] as String,
        score: (json['score'] as num).toDouble(),
      );
}

class Message {
  final String id;
  final MessageRole role;
  String content;
  final DateTime createdAt;
  bool streaming;
  // Чанки, на которых модель основывала ответ (если был RAG).
  List<MessageRagRef>? ragRefs;
  // Tool calls от assistant (agent mode).
  List<ToolCall>? toolCalls;
  // Tool result — связан с какой-то tool call.
  String? toolCallId;

  Message({
    String? id,
    required this.role,
    required this.content,
    DateTime? createdAt,
    this.streaming = false,
    this.ragRefs,
    this.toolCalls,
    this.toolCallId,
  })  : id = id ?? _uuid.v4(),
        createdAt = createdAt ?? DateTime.now();

  Map<String, dynamic> toOllamaJson() => {
        'role': _roleToString(role),
        'content': content,
      };

  Map<String, dynamic> toJson() => {
        'id': id,
        'role': _roleToString(role),
        'content': content,
        'createdAt': createdAt.toIso8601String(),
      };

  factory Message.fromJson(Map<String, dynamic> json) => Message(
        id: json['id'] as String?,
        role: _roleFromString(json['role'] as String),
        content: json['content'] as String,
        createdAt: DateTime.tryParse(json['createdAt'] as String? ?? ''),
      );
}

class Conversation {
  final String id;
  String title;
  String model;
  String systemPrompt;
  double temperature;
  final List<Message> messages;
  final DateTime createdAt;
  DateTime updatedAt;

  Conversation({
    String? id,
    this.title = 'New chat',
    required this.model,
    this.systemPrompt = '',
    this.temperature = 0.3,
    List<Message>? messages,
    DateTime? createdAt,
    DateTime? updatedAt,
  })  : id = id ?? _uuid.v4(),
        messages = messages ?? [],
        createdAt = createdAt ?? DateTime.now(),
        updatedAt = updatedAt ?? DateTime.now();

  void touch() => updatedAt = DateTime.now();

  void autoRename() {
    final firstUser = messages.firstWhere(
      (m) => m.role == MessageRole.user,
      orElse: () => Message(role: MessageRole.user, content: ''),
    );
    final text = firstUser.content.trim();
    if (text.isEmpty) return;
    final oneLine = text.replaceAll(RegExp(r'\s+'), ' ');
    title = oneLine.length > 40 ? '${oneLine.substring(0, 40)}…' : oneLine;
  }

  List<Map<String, dynamic>> toOllamaMessages() {
    final out = <Map<String, dynamic>>[];
    if (systemPrompt.trim().isNotEmpty) {
      out.add({'role': 'system', 'content': systemPrompt.trim()});
    }
    for (final m in messages) {
      if (m.streaming &&
          m.content.isEmpty &&
          (m.toolCalls?.isEmpty ?? true)) {
        continue;
      }
      final entry = <String, dynamic>{
        'role': _roleToString(m.role),
        'content': m.content,
      };
      if (m.toolCalls != null && m.toolCalls!.isNotEmpty) {
        entry['tool_calls'] =
            m.toolCalls!.map((c) => c.toOllamaFormat()).toList();
      }
      if (m.toolCallId != null) {
        entry['tool_call_id'] = m.toolCallId;
      }
      out.add(entry);
    }
    return out;
  }
}
