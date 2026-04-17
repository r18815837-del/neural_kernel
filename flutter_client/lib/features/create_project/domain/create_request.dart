/// Request model for project generation.
class CreateProjectRequest {
  final String text;
  final String outputFormat;
  final Map<String, dynamic>? metadata;

  const CreateProjectRequest({
    required this.text,
    this.outputFormat = 'zip',
    this.metadata,
  });

  Map<String, dynamic> toJson() {
    final json = <String, dynamic>{
      'text': text,
      'output_format': outputFormat,
    };
    if (metadata != null && metadata!.isNotEmpty) {
      json['metadata'] = metadata;
    }
    return json;
  }
}
