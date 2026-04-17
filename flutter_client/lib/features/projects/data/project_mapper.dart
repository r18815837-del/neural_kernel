import '../../../core/utils/date_utils.dart';
import '../domain/project.dart';
import 'project_dto.dart';

/// Maps backend DTOs to domain models.
class ProjectMapper {
  ProjectMapper._();

  static ProjectListItem toDomain(ProjectListItemDto dto) {
    return ProjectListItem(
      id: dto.projectId,
      name: dto.projectName,
      status: dto.status,
      statusLabel: dto.statusLabel,
      createdAt: NKDateUtils.tryParse(dto.createdAt),
      features: dto.features,
      artifactAvailable: dto.artifactAvailable,
    );
  }

  static List<ProjectListItem> toDomainList(ProjectListResponseDto dto) {
    return dto.projects.map(toDomain).toList();
  }
}
