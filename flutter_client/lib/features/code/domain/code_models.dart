/// Models for the code assistant feature.

class CodeAnalysisResult {
  final String? language;
  final bool syntaxValid;
  final List<Map<String, dynamic>> functions;
  final List<Map<String, dynamic>> classes;
  final List<String> imports;
  final List<String> issues;
  final List<String> suggestions;
  final String explanation;

  const CodeAnalysisResult({
    this.language,
    required this.syntaxValid,
    required this.functions,
    required this.classes,
    required this.imports,
    required this.issues,
    required this.suggestions,
    required this.explanation,
  });

  factory CodeAnalysisResult.fromJson(Map<String, dynamic> json) {
    return CodeAnalysisResult(
      language: json['language'] as String?,
      syntaxValid: json['syntax_valid'] as bool? ?? true,
      functions: (json['functions'] as List<dynamic>?)
              ?.map((e) => Map<String, dynamic>.from(e as Map))
              .toList() ??
          [],
      classes: (json['classes'] as List<dynamic>?)
              ?.map((e) => Map<String, dynamic>.from(e as Map))
              .toList() ??
          [],
      imports:
          (json['imports'] as List<dynamic>?)?.map((e) => e.toString()).toList() ??
              [],
      issues:
          (json['issues'] as List<dynamic>?)?.map((e) => e.toString()).toList() ??
              [],
      suggestions: (json['suggestions'] as List<dynamic>?)
              ?.map((e) => e.toString())
              .toList() ??
          [],
      explanation: json['explanation'] as String? ?? '',
    );
  }
}

class CodeRunResult {
  final String stdout;
  final String stderr;
  final bool success;
  final int exitCode;
  final bool timedOut;
  final String? errorSummary;

  const CodeRunResult({
    required this.stdout,
    required this.stderr,
    required this.success,
    required this.exitCode,
    required this.timedOut,
    this.errorSummary,
  });

  factory CodeRunResult.fromJson(Map<String, dynamic> json) {
    return CodeRunResult(
      stdout: json['stdout'] as String? ?? '',
      stderr: json['stderr'] as String? ?? '',
      success: json['success'] as bool? ?? false,
      exitCode: json['exit_code'] as int? ?? -1,
      timedOut: json['timed_out'] as bool? ?? false,
      errorSummary: json['error_summary'] as String?,
    );
  }
}

class CodeAskResult {
  final String answer;
  final double confidence;
  final String? language;
  final List<String> tips;

  const CodeAskResult({
    required this.answer,
    required this.confidence,
    this.language,
    required this.tips,
  });

  factory CodeAskResult.fromJson(Map<String, dynamic> json) {
    return CodeAskResult(
      answer: json['answer'] as String? ?? '',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      language: json['language'] as String?,
      tips: (json['tips'] as List<dynamic>?)
              ?.map((e) => e.toString())
              .toList() ??
          [],
    );
  }
}
