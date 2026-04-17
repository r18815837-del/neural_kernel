/// Tech stack info for the generated project.
class TechStack {
  final String? backend;
  final String? frontend;
  final String? database;
  final String? mobile;
  final String? deployment;

  const TechStack({
    this.backend,
    this.frontend,
    this.database,
    this.mobile,
    this.deployment,
  });

  /// Returns non-null entries as a map for display.
  Map<String, String> get entries {
    final map = <String, String>{};
    if (backend != null) map['Backend'] = backend!;
    if (frontend != null) map['Frontend'] = frontend!;
    if (database != null) map['Database'] = database!;
    if (mobile != null) map['Mobile'] = mobile!;
    if (deployment != null) map['Deployment'] = deployment!;
    return map;
  }

  bool get isEmpty => entries.isEmpty;
}
