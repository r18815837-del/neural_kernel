import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import 'tools.dart';

/// Одноходовой ответ (для agent-mode).
class ChatTurn {
  ChatTurn({required this.content, required this.toolCalls});
  final String content;
  final List<ToolCall> toolCalls;
}

/// Клиент к локальной Ollama. Поддерживает стриминг ответа.
class OllamaService {
  OllamaService({this.baseUrl = 'http://localhost:11434'});

  final String baseUrl;

  http.Client? _client;
  StreamSubscription<String>? _sub;

  Future<List<String>> listModels() async {
    final url = Uri.parse('$baseUrl/api/tags');
    try {
      final resp = await http.get(url).timeout(const Duration(seconds: 5));
      if (resp.statusCode != 200) {
        throw OllamaException('Ollama вернула ${resp.statusCode}: ${resp.body}');
      }
      final data = jsonDecode(resp.body) as Map<String, dynamic>;
      final models = (data['models'] as List? ?? [])
          .map((m) => (m as Map<String, dynamic>)['name'] as String)
          .toList()
        ..sort();
      return models;
    } on TimeoutException {
      throw OllamaException(
        'Ollama не отвечает на $baseUrl за 5 секунд. '
        'Проверь иконку ламы в трее.',
      );
    } catch (e) {
      throw OllamaException('Не удалось получить список моделей: $e');
    }
  }

  Future<void> chatStream({
    required String model,
    required List<Map<String, dynamic>> messages,
    required void Function(String token) onToken,
    required void Function() onDone,
    required void Function(Object error) onError,
    double temperature = 0.3,
    double topP = 0.9,
    int numCtx = 4096,
  }) async {
    await cancel();

    _client = http.Client();
    final url = Uri.parse('$baseUrl/api/chat');

    final request = http.Request('POST', url)
      ..headers['Content-Type'] = 'application/json'
      ..body = jsonEncode({
        'model': model,
        'messages': messages,
        'stream': true,
        'options': {
          'temperature': temperature,
          'top_p': topP,
          'num_ctx': numCtx,
        },
      });

    final completer = Completer<void>();

    try {
      final resp = await _client!.send(request);

      if (resp.statusCode != 200) {
        final body = await resp.stream.bytesToString();
        onError(OllamaException(
          'Ollama ${resp.statusCode}: ${body.isEmpty ? "(empty)" : body}',
        ));
        onDone();
        completer.complete();
        return completer.future;
      }

      _sub = resp.stream
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .listen(
        (line) {
          if (line.isEmpty) return;
          try {
            final data = jsonDecode(line) as Map<String, dynamic>;
            final msg = data['message'] as Map<String, dynamic>?;
            final content = msg?['content'] as String? ?? '';
            if (content.isNotEmpty) {
              onToken(content);
            }
            if (data['done'] == true) {
              onDone();
              if (!completer.isCompleted) completer.complete();
            }
          } catch (_) {
            // Битая строка — пропускаем.
          }
        },
        onError: (Object e, StackTrace st) {
          onError(e);
          onDone();
          if (!completer.isCompleted) completer.complete();
        },
        onDone: () {
          if (!completer.isCompleted) {
            onDone();
            completer.complete();
          }
        },
        cancelOnError: true,
      );
    } catch (e) {
      onError(e);
      onDone();
      if (!completer.isCompleted) completer.complete();
    }

    return completer.future;
  }

  /// Non-streaming chat (для agent mode с tools).
  /// Возвращает один ответ: текст + tool_calls.
  Future<ChatTurn> chat({
    required String model,
    required List<Map<String, dynamic>> messages,
    List<Map<String, dynamic>>? tools,
    double temperature = 0.3,
    double topP = 0.9,
    int numCtx = 4096,
  }) async {
    final url = Uri.parse('$baseUrl/api/chat');
    final body = <String, dynamic>{
      'model': model,
      'messages': messages,
      'stream': false,
      'options': {
        'temperature': temperature,
        'top_p': topP,
        'num_ctx': numCtx,
      },
    };
    if (tools != null && tools.isNotEmpty) {
      body['tools'] = tools;
    }

    final resp = await http
        .post(
          url,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(body),
        )
        .timeout(const Duration(minutes: 5));

    if (resp.statusCode != 200) {
      throw OllamaException(
        'Ollama ${resp.statusCode}: ${resp.body}',
      );
    }

    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    final msg = data['message'] as Map<String, dynamic>?;
    if (msg == null) {
      throw OllamaException('No message in response: ${resp.body}');
    }
    return ChatTurn(
      content: msg['content'] as String? ?? '',
      toolCalls: parseOllamaToolCalls(msg),
    );
  }

  Future<void> cancel() async {
    await _sub?.cancel();
    _sub = null;
    _client?.close();
    _client = null;
  }

  void dispose() {
    cancel();
  }
}

class OllamaException implements Exception {
  OllamaException(this.message);
  final String message;
  @override
  String toString() => message;
}
