import 'dart:convert';
import 'dart:io';

import 'package:glob/glob.dart';
import 'package:glob/list_local_fs.dart';
import 'package:path/path.dart' as p;
import 'package:uuid/uuid.dart';

const _uuid = Uuid();

// ══════════════════════════════════════════════════════════════════════
// Спецификации инструментов — в формате, ожидаемом Ollama / OpenAI.
// ══════════════════════════════════════════════════════════════════════
class ToolSpec {
  const ToolSpec({
    required this.name,
    required this.description,
    required this.parameters,
    this.requiresConfirmation = false,
  });

  final String name;
  final String description;
  final Map<String, dynamic> parameters; // JSON Schema
  final bool requiresConfirmation;

  Map<String, dynamic> toOllamaJson() => {
        'type': 'function',
        'function': {
          'name': name,
          'description': description,
          'parameters': parameters,
        },
      };
}

/// 6 базовых инструментов. Пишущие (write_file, edit_file) помечены
/// requiresConfirmation — executor спросит пользователя до исполнения.
const kToolSpecs = <ToolSpec>[
  ToolSpec(
    name: 'list_files',
    description:
        'List files and directories in a given path within the workspace. '
        'Use "." for the workspace root. Returns a newline-separated list '
        'with "/" suffix on directories.',
    parameters: {
      'type': 'object',
      'properties': {
        'path': {
          'type': 'string',
          'description':
              'Relative path within the workspace. Use "." for root.',
        }
      },
      'required': ['path'],
    },
  ),
  ToolSpec(
    name: 'read_file',
    description:
        'Read the full content of a file within the workspace. Returns text. '
        'Max 256 KB — larger files will fail.',
    parameters: {
      'type': 'object',
      'properties': {
        'path': {
          'type': 'string',
          'description': 'Relative path to the file.',
        }
      },
      'required': ['path'],
    },
  ),
  ToolSpec(
    name: 'glob_files',
    description:
        'Find files matching a glob pattern relative to the workspace. '
        'Examples: "**/*.py", "kernel/**/*.dart", "*.md". Returns up to '
        '200 paths.',
    parameters: {
      'type': 'object',
      'properties': {
        'pattern': {
          'type': 'string',
          'description': 'Glob pattern.',
        }
      },
      'required': ['pattern'],
    },
  ),
  ToolSpec(
    name: 'grep',
    description:
        'Search for a plain text or regex pattern in files within the '
        'workspace. Returns matches as "path:line:content" lines. '
        'Use glob to limit scope.',
    parameters: {
      'type': 'object',
      'properties': {
        'pattern': {
          'type': 'string',
          'description': 'Regex pattern to search for.',
        },
        'path_glob': {
          'type': 'string',
          'description':
              'Optional glob to restrict which files to search (e.g. "**/*.py"). '
                  'Default: all text files.',
        },
      },
      'required': ['pattern'],
    },
  ),
  ToolSpec(
    name: 'write_file',
    description:
        'Create or overwrite a file with the given content. REQUIRES USER '
        'CONFIRMATION before executing. Use sparingly and prefer edit_file '
        'for small changes.',
    parameters: {
      'type': 'object',
      'properties': {
        'path': {'type': 'string', 'description': 'Relative path.'},
        'content': {
          'type': 'string',
          'description': 'Full new content of the file.',
        },
      },
      'required': ['path', 'content'],
    },
    requiresConfirmation: true,
  ),
  ToolSpec(
    name: 'edit_file',
    description:
        'Replace a specific string in a file. old_string MUST match exactly '
        'one occurrence in the file — otherwise the edit fails. '
        'REQUIRES USER CONFIRMATION.',
    parameters: {
      'type': 'object',
      'properties': {
        'path': {'type': 'string'},
        'old_string': {
          'type': 'string',
          'description':
              'The exact text to find. Must appear exactly once in the file.',
        },
        'new_string': {
          'type': 'string',
          'description': 'Replacement text.',
        },
      },
      'required': ['path', 'old_string', 'new_string'],
    },
    requiresConfirmation: true,
  ),
];

// ══════════════════════════════════════════════════════════════════════
// Data classes
// ══════════════════════════════════════════════════════════════════════
class ToolCall {
  ToolCall({
    String? id,
    required this.name,
    required this.arguments,
  }) : id = id ?? _uuid.v4();

  final String id;
  final String name;
  final Map<String, dynamic> arguments;

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'arguments': arguments,
      };

  factory ToolCall.fromJson(Map<String, dynamic> json) => ToolCall(
        id: json['id'] as String?,
        name: json['name'] as String,
        arguments: Map<String, dynamic>.from(json['arguments'] as Map),
      );

  /// Для передачи обратно в Ollama.
  Map<String, dynamic> toOllamaFormat() => {
        'function': {
          'name': name,
          'arguments': arguments,
        },
      };
}

class ToolResult {
  ToolResult({
    required this.success,
    required this.content,
    this.rejected = false,
  });

  final bool success;
  final String content;
  final bool rejected;
}

class ToolsDisabledException implements Exception {
  ToolsDisabledException(this.reason);
  final String reason;
  @override
  String toString() => reason;
}

// ══════════════════════════════════════════════════════════════════════
// Executor
// ══════════════════════════════════════════════════════════════════════
class ToolExecutor {
  ToolExecutor({
    required this.workspaceRoot,
    required this.confirm,
  });

  /// Абсолютный путь к рабочей директории. Всё вне неё — запрещено.
  String workspaceRoot;

  /// Вызывается для инструментов с requiresConfirmation=true.
  /// Возвращает true если пользователь одобрил.
  Future<bool> Function(ToolCall) confirm;

  /// Максимальный размер файла для read (256 КБ).
  static const int _maxReadBytes = 256 * 1024;

  /// Не читаем/не listим эти папки — чтобы защитить проект и ОС.
  static const _skipParts = {
    '.git',
    '__pycache__',
    '.venv',
    '.venv-wsl',
    'venv',
    'env',
    'node_modules',
    'build',
    'dist',
    '.dart_tool',
    '.pytest_cache',
    '.idea',
    '.vscode',
    'qdrant_storage',
  };

  Future<ToolResult> execute(ToolCall call) async {
    if (workspaceRoot.isEmpty) {
      return ToolResult(
        success: false,
        content: 'Workspace not configured. Ask the user to pick a '
            'workspace folder in Settings.',
      );
    }

    // Проверяем существование workspace.
    final root = Directory(workspaceRoot);
    if (!root.existsSync()) {
      return ToolResult(
        success: false,
        content: 'Workspace directory does not exist: $workspaceRoot',
      );
    }

    final spec = kToolSpecs.firstWhere(
      (s) => s.name == call.name,
      orElse: () => const ToolSpec(
        name: '',
        description: '',
        parameters: {},
      ),
    );
    if (spec.name.isEmpty) {
      return ToolResult(
        success: false,
        content: 'Unknown tool: ${call.name}. Available: '
            '${kToolSpecs.map((s) => s.name).join(", ")}.',
      );
    }

    // Подтверждение для опасных.
    if (spec.requiresConfirmation) {
      final approved = await confirm(call);
      if (!approved) {
        return ToolResult(
          success: false,
          content: 'User rejected this ${call.name} operation.',
          rejected: true,
        );
      }
    }

    try {
      return await switch (call.name) {
        'list_files' => _listFiles(call),
        'read_file' => _readFile(call),
        'glob_files' => _globFiles(call),
        'grep' => _grep(call),
        'write_file' => _writeFile(call),
        'edit_file' => _editFile(call),
        _ => Future.value(ToolResult(
            success: false,
            content: 'Unhandled tool: ${call.name}',
          )),
      };
    } catch (e, st) {
      return ToolResult(
        success: false,
        content: 'Tool ${call.name} failed: $e\n${st.toString().split("\n").take(3).join("\n")}',
      );
    }
  }

  // ── Path safety ─────────────────────────────────────────────────────
  String _resolve(String relPath) {
    if (relPath.isEmpty) relPath = '.';
    // Windows-friendly normalization.
    final normalized = p.normalize(p.join(workspaceRoot, relPath));
    final rootAbs = p.normalize(workspaceRoot);
    // Должен быть внутри workspaceRoot ИЛИ равен ему.
    if (normalized != rootAbs && !p.isWithin(rootAbs, normalized)) {
      throw ToolsDisabledException(
          'Path escapes workspace: $relPath → $normalized');
    }
    return normalized;
  }

  // ── Tools ───────────────────────────────────────────────────────────
  Future<ToolResult> _listFiles(ToolCall call) async {
    final path = (call.arguments['path'] as String?) ?? '.';
    final abs = _resolve(path);
    final dir = Directory(abs);
    if (!dir.existsSync()) {
      return ToolResult(success: false, content: 'Not a directory: $path');
    }

    final entries = dir.listSync()
      ..sort((a, b) {
        final isDirA = a is Directory ? 0 : 1;
        final isDirB = b is Directory ? 0 : 1;
        if (isDirA != isDirB) return isDirA - isDirB;
        return p.basename(a.path).compareTo(p.basename(b.path));
      });

    final buf = StringBuffer();
    buf.writeln('Directory: ${p.relative(abs, from: workspaceRoot)}');
    for (final e in entries) {
      final name = p.basename(e.path);
      if (_skipParts.contains(name)) continue;
      if (e is Directory) {
        buf.writeln('  $name/');
      } else if (e is File) {
        final size = _formatBytes(e.lengthSync());
        buf.writeln('  $name  ($size)');
      }
    }
    return ToolResult(success: true, content: buf.toString());
  }

  Future<ToolResult> _readFile(ToolCall call) async {
    final path = call.arguments['path'] as String? ?? '';
    if (path.isEmpty) {
      return ToolResult(success: false, content: 'path is required');
    }
    final abs = _resolve(path);
    final file = File(abs);
    if (!file.existsSync()) {
      return ToolResult(success: false, content: 'File does not exist: $path');
    }
    final size = file.lengthSync();
    if (size > _maxReadBytes) {
      return ToolResult(
        success: false,
        content: 'File too large: ${_formatBytes(size)} > 256 KB. '
            'Use grep to find specific content.',
      );
    }
    final content = await file.readAsString();
    return ToolResult(
      success: true,
      content: content,
    );
  }

  Future<ToolResult> _globFiles(ToolCall call) async {
    final pattern = call.arguments['pattern'] as String? ?? '';
    if (pattern.isEmpty) {
      return ToolResult(success: false, content: 'pattern is required');
    }

    final matches = <String>[];
    try {
      final glob = Glob(pattern);
      await for (final entity in glob.list(root: workspaceRoot)) {
        if (entity is File) {
          final rel = p.relative(entity.path, from: workspaceRoot);
          // Фильтр тяжёлых папок.
          if (rel.split(p.separator).any(_skipParts.contains)) continue;
          matches.add(rel.replaceAll(p.separator, '/'));
          if (matches.length >= 200) break;
        }
      }
    } catch (e) {
      return ToolResult(success: false, content: 'Glob error: $e');
    }

    matches.sort();
    if (matches.isEmpty) {
      return ToolResult(success: true, content: 'No matches for: $pattern');
    }
    return ToolResult(
      success: true,
      content: '${matches.length} match(es):\n${matches.join("\n")}',
    );
  }

  Future<ToolResult> _grep(ToolCall call) async {
    final pattern = call.arguments['pattern'] as String? ?? '';
    final pathGlob = call.arguments['path_glob'] as String? ?? '**/*';
    if (pattern.isEmpty) {
      return ToolResult(success: false, content: 'pattern is required');
    }

    RegExp regex;
    try {
      regex = RegExp(pattern);
    } catch (e) {
      return ToolResult(success: false, content: 'Invalid regex: $e');
    }

    final glob = Glob(pathGlob);
    final results = <String>[];
    int filesScanned = 0;

    try {
      await for (final entity in glob.list(root: workspaceRoot)) {
        if (entity is! File) continue;
        final file = entity as File;
        final rel = p.relative(file.path, from: workspaceRoot);
        if (rel.split(p.separator).any(_skipParts.contains)) continue;

        // Пропускаем большие файлы и очевидно бинарные.
        final size = file.lengthSync();
        if (size > _maxReadBytes) continue;
        if (_looksBinary(file)) continue;

        filesScanned++;
        try {
          final text = await file.readAsString();
          final lines = text.split('\n');
          for (var i = 0; i < lines.length; i++) {
            if (regex.hasMatch(lines[i])) {
              final snippet = lines[i].length > 200
                  ? '${lines[i].substring(0, 200)}...'
                  : lines[i];
              results.add(
                  '${rel.replaceAll(p.separator, '/')}:${i + 1}: $snippet');
              if (results.length >= 200) break;
            }
          }
          if (results.length >= 200) break;
        } catch (_) {
          // Не смогли прочитать — пропускаем.
        }
      }
    } catch (e) {
      return ToolResult(success: false, content: 'Grep error: $e');
    }

    if (results.isEmpty) {
      return ToolResult(
        success: true,
        content: 'No matches. Scanned $filesScanned files.',
      );
    }
    return ToolResult(
      success: true,
      content:
          'Found ${results.length} match(es) (scanned $filesScanned files):\n'
          '${results.join("\n")}',
    );
  }

  Future<ToolResult> _writeFile(ToolCall call) async {
    final path = call.arguments['path'] as String? ?? '';
    final content = call.arguments['content'] as String? ?? '';
    if (path.isEmpty) {
      return ToolResult(success: false, content: 'path is required');
    }
    final abs = _resolve(path);
    final file = File(abs);
    await file.parent.create(recursive: true);
    await file.writeAsString(content);
    return ToolResult(
      success: true,
      content: 'Wrote ${content.length} bytes to $path',
    );
  }

  Future<ToolResult> _editFile(ToolCall call) async {
    final path = call.arguments['path'] as String? ?? '';
    final oldStr = call.arguments['old_string'] as String? ?? '';
    final newStr = call.arguments['new_string'] as String? ?? '';
    if (path.isEmpty || oldStr.isEmpty) {
      return ToolResult(
        success: false,
        content: 'path and old_string are required',
      );
    }
    final abs = _resolve(path);
    final file = File(abs);
    if (!file.existsSync()) {
      return ToolResult(success: false, content: 'File does not exist: $path');
    }
    final content = await file.readAsString();
    final count = oldStr.allMatches(content).length;
    if (count == 0) {
      return ToolResult(
        success: false,
        content: 'old_string not found in $path. Check exact text (whitespace, case).',
      );
    }
    if (count > 1) {
      return ToolResult(
        success: false,
        content: 'old_string matches $count times in $path — ambiguous. '
            'Provide more surrounding context to make it unique.',
      );
    }
    final newContent = content.replaceFirst(oldStr, newStr);
    await file.writeAsString(newContent);
    return ToolResult(
      success: true,
      content: 'Replaced 1 occurrence in $path '
          '(${oldStr.length}→${newStr.length} bytes)',
    );
  }

  // ── Helpers ─────────────────────────────────────────────────────────
  static String _formatBytes(int bytes) {
    if (bytes < 1024) return '${bytes}B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)}K';
    return '${(bytes / 1024 / 1024).toStringAsFixed(1)}M';
  }

  static bool _looksBinary(File f) {
    try {
      final head = f.openSync()..setPositionSync(0);
      final bytes = head.readSync(512);
      head.closeSync();
      for (final b in bytes) {
        if (b == 0) return true; // null byte ≈ binary
      }
      return false;
    } catch (_) {
      return true;
    }
  }
}

// ══════════════════════════════════════════════════════════════════════
// Ollama tool_call parsing — извлечь ToolCall из ответа модели.
// ══════════════════════════════════════════════════════════════════════
List<ToolCall> parseOllamaToolCalls(Map<String, dynamic> messageJson) {
  final raw = messageJson['tool_calls'];
  if (raw is! List || raw.isEmpty) return const [];

  final out = <ToolCall>[];
  for (final item in raw) {
    if (item is! Map<String, dynamic>) continue;
    final fn = item['function'];
    if (fn is! Map<String, dynamic>) continue;
    final name = fn['name'] as String?;
    if (name == null || name.isEmpty) continue;

    // arguments может прийти как Map или как JSON-строка.
    Map<String, dynamic> args = {};
    final rawArgs = fn['arguments'];
    if (rawArgs is Map) {
      args = Map<String, dynamic>.from(rawArgs);
    } else if (rawArgs is String) {
      try {
        final parsed = jsonDecode(rawArgs);
        if (parsed is Map) args = Map<String, dynamic>.from(parsed);
      } catch (_) {}
    }

    out.add(ToolCall(
      id: item['id'] as String? ?? _uuid.v4(),
      name: name,
      arguments: args,
    ));
  }
  return out;
}
