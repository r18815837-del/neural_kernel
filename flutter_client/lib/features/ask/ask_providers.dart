import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/api/api_client.dart';
import 'data/ask_api.dart';
import 'domain/ask_models.dart';

/// Ask API instance.
final askApiProvider = Provider<AskApi>((ref) {
  return AskApi(ref.watch(apiClientProvider));
});

/// State for the ask screen.
final askResultProvider =
    StateProvider<AsyncValue<AskResult>?>((ref) => null);
