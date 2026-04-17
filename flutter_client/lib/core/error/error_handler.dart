import '../api/api_exceptions.dart';
import 'error_models.dart';

/// Convert any exception to [AppError].
AppError handleError(Object error) {
  if (error is ApiException) {
    return AppError.from(error);
  }
  return AppError.unexpected(error);
}
