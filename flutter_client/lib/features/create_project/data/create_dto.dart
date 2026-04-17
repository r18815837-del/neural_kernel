/// Response DTO from POST /generate.
class CreateProjectResponseDto {
  final String projectId;
  final String status;
  final String? message;

  CreateProjectResponseDto({
    required this.projectId,
    required this.status,
    this.message,
  });

  factory CreateProjectResponseDto.fromJson(Map<String, dynamic> json) {
    return CreateProjectResponseDto(
      projectId: json['project_id'] as String? ?? '',
      status: json['status'] as String? ?? 'pending',
      message: json['message'] as String?,
    );
  }
}
