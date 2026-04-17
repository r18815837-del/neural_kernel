import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/api/api_client.dart';
import '../data/create_api.dart';
import '../domain/create_request.dart';

/// Create API instance.
final createApiProvider = Provider<CreateApi>((ref) {
  return CreateApi(ref.watch(apiClientProvider));
});

/// State for the create screen.
enum CreateStatus { idle, submitting, success, error }

class CreateState {
  final CreateStatus status;
  final String? projectId;
  final String? error;

  const CreateState({
    this.status = CreateStatus.idle,
    this.projectId,
    this.error,
  });
}

final createProjectProvider =
    AutoDisposeNotifierProvider<CreateProjectNotifier, CreateState>(
  CreateProjectNotifier.new,
);

class CreateProjectNotifier extends AutoDisposeNotifier<CreateState> {
  @override
  CreateState build() => const CreateState();

  Future<void> submit(String text, {String outputFormat = 'zip'}) async {
    state = const CreateState(status: CreateStatus.submitting);

    try {
      final api = ref.read(createApiProvider);
      final request = CreateProjectRequest(
        text: text,
        outputFormat: outputFormat,
      );
      final resp = await api.generate(request);
      state = CreateState(
        status: CreateStatus.success,
        projectId: resp.projectId,
      );
    } catch (e) {
      state = CreateState(
        status: CreateStatus.error,
        error: e.toString(),
      );
    }
  }
}
