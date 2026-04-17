/// All backend URL paths in one place.
///
/// Usage: `ApiEndpoints.clientStatus('proj-123')` → `/api/v1/client/status/proj-123`
class ApiEndpoints {
  ApiEndpoints._();

  // Health
  static const health = '/api/v1/health';

  // Generation
  static const generate = '/api/v1/generate';

  // Client-facing
  static const clientProjects = '/api/v1/client/projects';
  static String clientStatus(String id) => '/api/v1/client/status/$id';
  static String clientDownloadInfo(String id) =>
      '/api/v1/client/download/$id/info';

  // Download (binary)
  static String downloadArtifact(String id) => '/api/v1/download/$id';

  // Cognition
  static const ask = '/api/v1/ask';

  // Code assistant
  static const codeAnalyze = '/api/v1/code/analyze';
  static const codeRun = '/api/v1/code/run';
  static const codeAsk = '/api/v1/code/ask';

  // Sessions
  static const sessions = '/api/v1/sessions';
  static String session(String id) => '/api/v1/sessions/$id';
  static String sessionMessages(String id) => '/api/v1/sessions/$id/messages';

  // Lifecycle
  static String transitions(String id) =>
      '/api/v1/projects/$id/transitions';
  static String archive(String id) => '/api/v1/projects/$id/archive';
  static String retry(String id) => '/api/v1/projects/$id/retry';
  static String transition(String id) =>
      '/api/v1/projects/$id/transition';
  static String versions(String id) => '/api/v1/projects/$id/versions';
  static String retain(String id) => '/api/v1/projects/$id/retain';
}
