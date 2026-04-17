import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/api/api_client.dart';
import 'data/project_mapper.dart';
import 'data/projects_api.dart';
import 'domain/project.dart';

/// API instance.
final projectsApiProvider = Provider<ProjectsApi>((ref) {
  return ProjectsApi(ref.watch(apiClientProvider));
});

/// The project list — fetch once, pull-to-refresh.
final projectListProvider =
    AsyncNotifierProvider<ProjectListNotifier, List<ProjectListItem>>(
  ProjectListNotifier.new,
);

class ProjectListNotifier extends AsyncNotifier<List<ProjectListItem>> {
  @override
  Future<List<ProjectListItem>> build() => _fetch();

  Future<List<ProjectListItem>> _fetch() async {
    final api = ref.read(projectsApiProvider);
    final dto = await api.getProjects();
    return ProjectMapper.toDomainList(dto);
  }

  /// Pull-to-refresh.
  Future<void> refresh() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(_fetch);
  }
}
