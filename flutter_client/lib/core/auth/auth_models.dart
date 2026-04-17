/// Auth mode — how the client authenticates with the backend.
enum AuthMode {
  none,
  apiKey,
  bearer;

  String get label => switch (this) {
        none => 'No Auth',
        apiKey => 'API Key',
        bearer => 'Bearer Token',
      };
}

/// Persisted credentials.
class AuthCredentials {
  final String baseUrl;
  final AuthMode mode;
  final String secret; // API key or JWT token (empty for none)

  const AuthCredentials({
    required this.baseUrl,
    required this.mode,
    this.secret = '',
  });

  bool get isValid =>
      baseUrl.isNotEmpty &&
      (mode == AuthMode.none || secret.isNotEmpty);
}

/// Auth state visible to the rest of the app.
enum AuthStatus {
  unknown,       // app just launched, checking storage
  unauthenticated,
  authenticated,
}

class AuthState {
  final AuthStatus status;
  final AuthCredentials? credentials;

  const AuthState({
    this.status = AuthStatus.unknown,
    this.credentials,
  });

  const AuthState.unknown()
      : status = AuthStatus.unknown,
        credentials = null;

  const AuthState.unauthenticated()
      : status = AuthStatus.unauthenticated,
        credentials = null;

  AuthState.authenticated(AuthCredentials creds)
      : status = AuthStatus.authenticated,
        credentials = creds;

  bool get isAuthenticated => status == AuthStatus.authenticated;
}
