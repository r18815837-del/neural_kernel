import 'package:shared_preferences/shared_preferences.dart';

import 'auth_models.dart';

/// Persistence for auth credentials via SharedPreferences.
class AuthStorage {
  static const _keyBaseUrl = 'nk_base_url';
  static const _keyAuthMode = 'nk_auth_mode';
  static const _keySecret = 'nk_secret';

  Future<AuthCredentials?> load() async {
    final prefs = await SharedPreferences.getInstance();
    final baseUrl = prefs.getString(_keyBaseUrl);
    final modeStr = prefs.getString(_keyAuthMode);
    final secret = prefs.getString(_keySecret);

    if (baseUrl == null || modeStr == null) return null;

    final AuthMode mode;
    switch (modeStr) {
      case 'none':
        mode = AuthMode.none;
      case 'bearer':
        mode = AuthMode.bearer;
      default:
        mode = AuthMode.apiKey;
    }
    final creds = AuthCredentials(
      baseUrl: baseUrl,
      mode: mode,
      secret: secret ?? '',
    );
    return creds.isValid ? creds : null;
  }

  Future<void> save(AuthCredentials credentials) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyBaseUrl, credentials.baseUrl);
    await prefs.setString(
      _keyAuthMode,
      credentials.mode.name,
    );
    await prefs.setString(_keySecret, credentials.secret);
  }

  Future<void> clear() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_keyBaseUrl);
    await prefs.remove(_keyAuthMode);
    await prefs.remove(_keySecret);
  }

  Future<bool> hasCredentials() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.containsKey(_keyBaseUrl) &&
        prefs.containsKey(_keySecret);
  }
}
