/// Quality metrics for the generated project.
class QualityScore {
  final bool? scaffoldValid;
  final bool? executionReady;
  final bool? consistencyOk;
  final double? overallScore;

  const QualityScore({
    this.scaffoldValid,
    this.executionReady,
    this.consistencyOk,
    this.overallScore,
  });

  /// 0-100 integer for display.
  int? get overallPercent =>
      overallScore != null ? (overallScore! * 100).round() : null;
}
