// Typed API exceptions thrown by the error interceptor.
// These are caught in providers / screens and converted to [AppError].

class ApiException implements Exception {
  final int? statusCode;
  final String message;
  final String? code;
  final bool retryable;
  final Map<String, dynamic>? data;

  const ApiException({
    this.statusCode,
    this.message = 'Unknown API error',
    this.code,
    this.retryable = false,
    this.data,
  });

  @override
  String toString() => 'ApiException($statusCode, $code: $message)';
}

class NetworkException extends ApiException {
  const NetworkException([String message = 'Network unavailable'])
      : super(message: message, retryable: true);
}

class TimeoutException extends ApiException {
  const TimeoutException([String message = 'Request timed out'])
      : super(message: message, retryable: true);
}

class UnauthorizedException extends ApiException {
  const UnauthorizedException([String message = 'Session expired'])
      : super(statusCode: 401, message: message, code: 'unauthorized');
}

class ForbiddenException extends ApiException {
  const ForbiddenException([String message = 'Access denied'])
      : super(statusCode: 403, message: message, code: 'forbidden');
}

class NotFoundException extends ApiException {
  const NotFoundException([String message = 'Not found'])
      : super(
          statusCode: 404,
          message: message,
          code: 'not_found',
          retryable: false,
        );
}

class ConflictException extends ApiException {
  const ConflictException([String message = 'Conflict'])
      : super(statusCode: 409, message: message, code: 'conflict');
}

class ServerException extends ApiException {
  const ServerException([String message = 'Server error'])
      : super(statusCode: 500, message: message, retryable: true);
}

class BadRequestException extends ApiException {
  const BadRequestException(String message, {Map<String, dynamic>? data}) // ignore: use_super_parameters
      : super(
          statusCode: 400,
          message: message,
          code: 'bad_request',
          data: data,
        );
}
