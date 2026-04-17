import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/api/api_client.dart';
import '../../core/polling/polling_controller.dart';
import 'data/status_api.dart';
import 'data/status_mapper.dart';
import 'domain/project_status.dart';

/// Status API instance.
final statusApiProvider = Provider<StatusApi>((ref) {
  return StatusApi(ref.watch(apiClientProvider));
});

/// Polling stream for a specific project — auto-disposes when leaving screen.
final projectStatusStreamProvider =
    StreamProvider.autoDispose.family<ProjectStatus, String>(
  (ref, projectId) {
    final api = ref.watch(statusApiProvider);

    return PollingController.poll<ProjectStatus>(
      fetcher: () async {
        final dto = await api.getStatus(projectId);
        return StatusMapper.toDomain(dto);
      },
      isDone: (s) => s.isTerminal,
      interval: (s) => s.pollInterval,
    );
  },
);
