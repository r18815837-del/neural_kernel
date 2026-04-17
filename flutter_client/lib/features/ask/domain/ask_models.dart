/// Models for the cognition ask feature.

class AskResult {
  final String id;
  final String question;
  final String answer;
  final String confidence;
  final List<String> sources;
  final List<StepInfo> steps;
  final String? error;

  const AskResult({
    required this.id,
    required this.question,
    required this.answer,
    required this.confidence,
    required this.sources,
    required this.steps,
    this.error,
  });

  factory AskResult.fromJson(Map<String, dynamic> json) {
    return AskResult(
      id: json['id'] as String? ?? '',
      question: json['question'] as String? ?? '',
      answer: json['answer'] as String? ?? '',
      confidence: json['confidence'] as String? ?? 'none',
      sources: (json['sources'] as List<dynamic>?)
              ?.map((e) => e.toString())
              .toList() ??
          [],
      steps: (json['steps'] as List<dynamic>?)
              ?.map(
                (e) => StepInfo.fromJson(e as Map<String, dynamic>),
              )
              .toList() ??
          [],
      error: json['error'] as String?,
    );
  }
}

class StepInfo {
  final String kind;
  final String output;
  final double? durationMs;

  const StepInfo({
    required this.kind,
    required this.output,
    this.durationMs,
  });

  factory StepInfo.fromJson(Map<String, dynamic> json) {
    return StepInfo(
      kind: json['kind'] as String? ?? '',
      output: json['output'] as String? ?? '',
      durationMs: (json['duration_ms'] as num?)?.toDouble(),
    );
  }
}
