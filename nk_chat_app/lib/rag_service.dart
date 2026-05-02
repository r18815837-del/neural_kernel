import 'dart:convert';

import 'package:http/http.dart' as http;

/// Клиент к локальному Qdrant + Ollama embeddings для code RAG.
///
/// Используется, когда пользователь включил «Use codebase context»:
///   1. Берём текст вопроса.
///   2. Считаем его embedding через Ollama nomic-embed-text.
///   3. Ищем top-K похожих чанков в Qdrant (коллекция neural_kernel_code).
///   4. Склеиваем в блок и возвращаем для вставки в system prompt.
class RagService {
  RagService({
    this.qdrantUrl = 'http://localhost:6333',
    this.ollamaUrl = 'http://localhost:11434',
    this.collection = 'neural_kernel_code',
    this.embedModel = 'nomic-embed-text',
  });

  final String qdrantUrl;
  final String ollamaUrl;
  final String collection;
  final String embedModel;

  /// Проверка что сервис живой. Возвращает null если всё ок,
  /// или строку-ошибку.
  Future<String?> ping() async {
    try {
      final r1 = await http
          .get(Uri.parse('$qdrantUrl/collections/$collection'))
          .timeout(const Duration(seconds: 3));
      if (r1.statusCode == 404) {
        return 'Qdrant коллекция "$collection" не найдена. '
            'Запусти: python rag/index_codebase.py';
      }
      if (r1.statusCode != 200) {
        return 'Qdrant вернул ${r1.statusCode}';
      }
    } catch (e) {
      return 'Qdrant недоступен ($qdrantUrl): $e';
    }

    try {
      final r2 = await http
          .get(Uri.parse('$ollamaUrl/api/tags'))
          .timeout(const Duration(seconds: 3));
      if (r2.statusCode != 200) return 'Ollama вернула ${r2.statusCode}';
    } catch (e) {
      return 'Ollama недоступна: $e';
    }

    return null;
  }

  /// Ищет релевантные куски кода по текстовому запросу.
  Future<List<CodeChunk>> search(String query, {int limit = 5}) async {
    // 1. Получаем embedding вопроса.
    final embedResp = await http
        .post(
          Uri.parse('$ollamaUrl/api/embeddings'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'model': embedModel, 'prompt': query}),
        )
        .timeout(const Duration(seconds: 15));

    if (embedResp.statusCode != 200) {
      throw Exception('Ollama embedding: ${embedResp.statusCode}');
    }

    final embedding = (jsonDecode(embedResp.body) as Map)['embedding'] as List;

    // 2. Ищем в Qdrant.
    final searchResp = await http
        .post(
          Uri.parse('$qdrantUrl/collections/$collection/points/search'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'vector': embedding,
            'limit': limit,
            'with_payload': true,
            'with_vector': false,
          }),
        )
        .timeout(const Duration(seconds: 10));

    if (searchResp.statusCode != 200) {
      throw Exception('Qdrant search: ${searchResp.statusCode}');
    }

    final hits = (jsonDecode(searchResp.body) as Map)['result'] as List;
    return hits.map((h) => CodeChunk.fromQdrant(h as Map<String, dynamic>)).toList();
  }

  /// Собирает найденные чанки в один markdown-блок, готовый к вставке
  /// в system prompt. Если список пуст — возвращает null.
  String? buildContextBlock(List<CodeChunk> chunks) {
    if (chunks.isEmpty) return null;
    final buf = StringBuffer();
    buf.writeln('# Relevant context from the user\'s neural_kernel codebase');
    buf.writeln();
    buf.writeln('The following snippets were retrieved semantically to help '
        'answer the user\'s question. Cite files by path when relevant.');
    buf.writeln();
    for (final c in chunks) {
      buf.writeln('## `${c.file}` (${c.kind} `${c.name}`, '
          'lines ${c.startLine}-${c.endLine})');
      buf.writeln('```python');
      buf.writeln(c.text.trim());
      buf.writeln('```');
      buf.writeln();
    }
    return buf.toString();
  }
}

class CodeChunk {
  CodeChunk({
    required this.file,
    required this.startLine,
    required this.endLine,
    required this.kind,
    required this.name,
    required this.text,
    required this.score,
  });

  final String file;
  final int startLine;
  final int endLine;
  final String kind;
  final String name;
  final String text;
  final double score;

  factory CodeChunk.fromQdrant(Map<String, dynamic> hit) {
    final payload = hit['payload'] as Map<String, dynamic>? ?? {};
    return CodeChunk(
      file: payload['file'] as String? ?? '?',
      startLine: (payload['start_line'] as num?)?.toInt() ?? 0,
      endLine: (payload['end_line'] as num?)?.toInt() ?? 0,
      kind: payload['kind'] as String? ?? 'chunk',
      name: payload['name'] as String? ?? '?',
      text: payload['text'] as String? ?? '',
      score: (hit['score'] as num?)?.toDouble() ?? 0.0,
    );
  }
}
