import '../api/api_exceptions.dart';

/// Central error classification visible to the UI layer.
enum AppErrorType {
  network,
  unauthorized,
  forbidden,
  notFound,
  conflict,
  serverError,
  generationFailed,
  validationError,
  unknown,
}

/// An error that the UI layer can display to the user.
class AppError {
  final AppErrorType type;
  final String userMessage;
  final String? technicalDetail;
  final bool retryable;
  final String? errorCode;

  const AppError({
    required this.type,
    required this.userMessage,
    this.technicalDetail,
    this.retryable = false,
    this.errorCode,
  });

  /// Map an [ApiException] to an [AppError] for the UI.
  factory AppError.from(ApiException e) {
    final type = switch (e) {
      UnauthorizedException() => AppErrorType.unauthorized,
      ForbiddenException() => AppErrorType.forbidden,
      NotFoundException() => AppErrorType.notFound,
      ConflictException() => AppErrorType.conflict,
      NetworkException() => AppErrorType.network,
      TimeoutException() => AppErrorType.network,
      ServerException() => AppErrorType.serverError,
      BadRequestException() => AppErrorType.validationError,
      _ => AppErrorType.unknown,
    };
    return AppError(
      type: type,
      userMessage: _userMessage(type, e.message),
      technicalDetail: e.toString(),
      retryable: e.retryable,
      errorCode: e.code,
    );
  }

  /// Catch-all for non-API exceptions.
  factory AppError.unexpected(Object error) {
    return AppError(
      type: AppErrorType.unknown,
      userMessage: 'Something went wrong',
      technicalDetail: error.toString(),
      retryable: true,
    );
  }

  static String _userMessage(AppErrorType type, String raw) {
    return switch (type) {
      AppErrorType.network => 'No internet connection',
      AppErrorType.unauthorized => 'Session expired. Please reconnect.',
      AppErrorType.forbidden => 'Access denied',
      AppErrorType.notFound => 'Not found',
      AppErrorType.conflict => raw,
      AppErrorType.serverError => 'Server error. Please try again.',
      AppErrorType.generationFailed => raw,
      AppErrorType.validationError => raw,
      AppErrorType.unknown => 'Something went wrong',
    };
  }
}
