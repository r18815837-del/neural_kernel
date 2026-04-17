# Neural Kernel — Flutter Client Architecture Blueprint

## 1. Архитектура: общий подход

Приложение строится на **feature-first Clean Architecture** с чётким разделением на три слоя:

- **Data** — API client, JSON-DTO, mappers. Знает о HTTP, JSON, хедерах. Ничего не знает о виджетах.
- **Domain** — app models, repositories (абстракции), use cases (если нужны). Чистый Dart, без зависимостей от Flutter/HTTP.
- **Presentation** — screens, widgets, state (Riverpod providers). Знает о domain models, ничего не знает о JSON.

Каждая feature содержит все три слоя внутри себя. Общий код (HTTP client, auth, error handling) живёт в `core/`.

State management: **Riverpod** (обоснование в секции 3).

---

## 2. Структура проекта

```
lib/
├── main.dart                          # entry point, ProviderScope, router
├── app/
│   ├── app.dart                       # MaterialApp + GoRouter setup
│   ├── router.dart                    # все routes, guards
│   └── theme.dart                     # тема, цвета, typography
│
├── core/
│   ├── api/
│   │   ├── api_client.dart            # Dio instance, base config
│   │   ├── api_interceptors.dart      # auth header, error transform, logging
│   │   ├── api_endpoints.dart         # const strings всех URL paths
│   │   └── api_exceptions.dart        # typed exceptions: ApiException, NetworkException
│   ├── auth/
│   │   ├── auth_provider.dart         # Riverpod: authStateProvider, tokenProvider
│   │   ├── auth_service.dart          # save/load/clear token, check validity
│   │   ├── auth_models.dart           # AuthCredentials, AuthMode enum
│   │   └── auth_storage.dart          # secure storage wrapper (flutter_secure_storage)
│   ├── error/
│   │   ├── error_handler.dart         # central error → user message mapping
│   │   ├── error_models.dart          # AppError, AppErrorType enum
│   │   └── error_widgets.dart         # ErrorBanner, RetryButton, ErrorDialog
│   ├── polling/
│   │   ├── polling_controller.dart    # generic poll-until-done с backoff
│   │   └── polling_config.dart        # intervals, max retries
│   └── utils/
│       ├── date_utils.dart            # ISO8601 parsing, relative time
│       ├── file_utils.dart            # download path, share file
│       └── extensions.dart            # String, BuildContext extensions
│
├── features/
│   ├── auth/
│   │   ├── data/
│   │   │   └── auth_api.dart          # POST /auth/token (future), validate key
│   │   ├── presentation/
│   │   │   ├── connect_screen.dart    # ввод URL + API key / JWT token
│   │   │   └── connect_form.dart      # form widget
│   │   └── auth_providers.dart        # feature-level providers
│   │
│   ├── projects/
│   │   ├── data/
│   │   │   ├── projects_api.dart      # GET /client/projects
│   │   │   ├── project_dto.dart       # raw JSON → Dart class
│   │   │   └── project_mapper.dart    # DTO → domain model
│   │   ├── domain/
│   │   │   ├── project.dart           # ProjectListItem model
│   │   │   └── project_repository.dart # abstract repo (for testability)
│   │   ├── presentation/
│   │   │   ├── projects_screen.dart   # список проектов
│   │   │   ├── project_card.dart      # карточка в списке
│   │   │   └── projects_providers.dart
│   │   └── projects_providers.dart    # top-level feature providers
│   │
│   ├── create_project/
│   │   ├── data/
│   │   │   ├── create_api.dart        # POST /generate
│   │   │   └── create_dto.dart
│   │   ├── presentation/
│   │   │   ├── create_screen.dart     # form: текст запроса + options
│   │   │   └── create_providers.dart
│   │   └── domain/
│   │       └── create_request.dart    # typed request model
│   │
│   ├── project_status/
│   │   ├── data/
│   │   │   ├── status_api.dart        # GET /client/status/{id}
│   │   │   ├── status_dto.dart        # ProjectStatusDTO mapping
│   │   │   └── status_mapper.dart
│   │   ├── domain/
│   │   │   ├── project_status.dart    # rich domain model
│   │   │   ├── feature_item.dart
│   │   │   ├── tech_stack.dart
│   │   │   └── quality_score.dart
│   │   ├── presentation/
│   │   │   ├── status_screen.dart     # main detail screen
│   │   │   ├── progress_section.dart  # progress bar + percent
│   │   │   ├── quality_section.dart   # scaffold_valid, exec_ready, consistency
│   │   │   ├── features_section.dart  # feature chips/list
│   │   │   ├── tech_stack_section.dart
│   │   │   ├── pipeline_section.dart  # agent count, llm used
│   │   │   ├── artifact_section.dart  # download button, metadata preview
│   │   │   └── status_providers.dart  # polling provider
│   │   └── project_status_providers.dart
│   │
│   ├── artifact/
│   │   ├── data/
│   │   │   ├── artifact_api.dart      # GET /client/download/{id}/info, GET /download/{id}
│   │   │   ├── artifact_dto.dart
│   │   │   └── download_service.dart  # Dio download + progress tracking
│   │   ├── domain/
│   │   │   └── artifact_metadata.dart
│   │   ├── presentation/
│   │   │   ├── artifact_screen.dart   # полная metadata + download
│   │   │   ├── download_button.dart   # progress indicator inside button
│   │   │   └── artifact_providers.dart
│   │   └── artifact_providers.dart
│   │
│   ├── lifecycle/
│   │   ├── data/
│   │   │   ├── lifecycle_api.dart     # POST archive/retry/transition, GET versions/transitions
│   │   │   └── lifecycle_dto.dart
│   │   ├── domain/
│   │   │   ├── lifecycle_state.dart
│   │   │   └── version_info.dart
│   │   ├── presentation/
│   │   │   ├── lifecycle_sheet.dart   # bottom sheet: archive, retry, allowed transitions
│   │   │   ├── versions_screen.dart   # список версий артефактов
│   │   │   └── lifecycle_providers.dart
│   │   └── lifecycle_providers.dart
│   │
│   └── settings/
│       └── presentation/
│           ├── settings_screen.dart   # URL, auth mode, debug toggle, logout
│           └── settings_providers.dart
│
└── shared/
    ├── widgets/
    │   ├── nk_app_bar.dart            # стандартный app bar
    │   ├── nk_loading.dart            # shimmer / skeleton
    │   ├── nk_empty_state.dart        # "Нет проектов" placeholder
    │   ├── nk_status_badge.dart       # цветной badge для статуса
    │   ├── nk_error_view.dart         # full-screen error with retry
    │   └── nk_chip.dart              # для features / tech_stack
    └── constants/
        ├── status_colors.dart         # pending=orange, completed=green, failed=red
        └── strings.dart               # hardcoded strings (до l10n)
```

---

## 3. State management: Riverpod

**Выбор: Riverpod 2.x** (с `riverpod_annotation` code generation).

Почему не Bloc: Bloc хорош для complex event-driven UI, но для этого проекта основной паттерн — это **async data fetching + polling + caching**, и Riverpod справляется с этим элегантнее. Меньше boilerplate, нативная работа с `AsyncValue`, автоматический dispose, declarative dependencies между providers.

Почему не Provider: Provider — это subset Riverpod. Riverpod безопаснее (compile-time проверки, нет `context`-зависимости), лучше тестируется.

### Ключевые providers

```
authStateProvider        → AsyncNotifier<AuthState>        — credentials, isAuthenticated
apiClientProvider        → Provider<Dio>                   — configured Dio instance
projectListProvider      → AsyncNotifier<List<Project>>    — fetchable, refreshable list
projectStatusProvider(id)→ StreamProvider<ProjectStatus>   — polling stream для конкретного проекта
createProjectProvider    → AsyncNotifier<CreateState>      — form state + submit
artifactMetaProvider(id) → FutureProvider<ArtifactMeta>    — artifact info
downloadProvider(id)     → StreamNotifier<DownloadState>   — progress 0.0-1.0
lifecycleProvider(id)    → AsyncNotifier<LifecycleState>   — allowed transitions, actions
settingsProvider         → Notifier<AppSettings>           — URL, auth mode, debug
```

### Паттерн polling через Riverpod

```dart
// status_providers.dart
@riverpod
Stream<ProjectStatus> projectStatusStream(ref, String projectId) async* {
  final api = ref.watch(statusApiProvider);
  while (true) {
    final status = await api.getStatus(projectId);
    yield status;
    if (status.isTerminal) break;  // completed | failed | archived
    await Future.delayed(status.pollInterval);
  }
}
```

`StreamProvider` автоматически отменяет stream при уходе со screen.

---

## 4. Features (список)

| Feature | Описание | Приоритет |
|---------|----------|-----------|
| `auth` | Подключение к backend, хранение credentials | MVP |
| `projects` | Список проектов пользователя | MVP |
| `create_project` | Создание проекта по текстовому запросу | MVP |
| `project_status` | Детальный статус, progress, quality, features, tech stack | MVP |
| `artifact` | Metadata артефакта, download, share | MVP |
| `lifecycle` | Archive, retry, transitions, versions | Phase 2 |
| `settings` | API URL, auth mode, debug, logout | Phase 2 |

---

## 5. Экраны

### 5.1 ConnectScreen (auth)
**Путь:** `/connect`
**Назначение:** первоначальное подключение к backend.

Содержит:
- Поле: Base URL (e.g. `https://api.neural-kernel.dev`)
- Toggle: API Key / Bearer Token
- Поле: ключ или токен
- Кнопка "Connect" → валидация через `GET /api/v1/health` + test auth через `GET /api/v1/client/projects`
- Сохранение в secure storage
- При успехе → redirect на ProjectsScreen

**Guard:** если credentials сохранены и валидны → skip, сразу на projects.

### 5.2 ProjectsScreen
**Путь:** `/projects`
**Назначение:** главный список проектов.

Содержит:
- Pull-to-refresh
- Список карточек, каждая показывает: project_name, status badge, created_at (relative), features chips (max 3), artifact available indicator
- FAB "+" → navigate to CreateScreen
- Tap на карточку → navigate to StatusScreen
- Long press → quick actions bottom sheet (archive, retry)
- Empty state: "Нет проектов. Создайте первый!"
- Shimmer loading state

### 5.3 CreateProjectScreen
**Путь:** `/create`
**Назначение:** ввод запроса и запуск генерации.

Содержит:
- TextField: текстовый запрос (multiline, min 10 chars)
- Optional expandable: output_format selector (zip/folder)
- Optional expandable: metadata key-value pairs
- Кнопка "Generate" → POST /generate → получаем project_id → navigate to StatusScreen

### 5.4 ProjectStatusScreen
**Путь:** `/projects/:id`
**Назначение:** центральный экран — всё о проекте.

Секции (scrollable):
1. **Header:** project_name, status badge, progress bar (0-100%)
2. **Message:** status_label + message text
3. **Quality:** scaffold_valid, execution_ready, consistency_ok, overall_score (color-coded)
4. **Features:** chip list с priority badges
5. **Tech Stack:** backend/frontend/database/mobile/deployment (только non-null)
6. **Pipeline:** agent_count, successful/failed counts, llm_used badge
7. **Artifact:** кнопка download (если artifact_available), size, format
8. **Error:** red banner если status=failed
9. **Actions:** AppBar actions → archive, retry (если allowed)

**Polling:** при status in (pending, in_progress) → stream provider polls каждые 2-5s с backoff.
При completed/failed → stop polling, show final state.

### 5.5 ArtifactScreen
**Путь:** `/projects/:id/artifact`
**Назначение:** детальная metadata + download.

Содержит:
- artifact_name, size (human-readable), format, version
- Download button с progress bar
- Кнопки: Open (если zip viewer есть), Share
- Features list (полная)
- Tech stack details

### 5.6 VersionsScreen (Phase 2)
**Путь:** `/projects/:id/versions`
**Назначение:** список версий артефакта.

Содержит:
- Timeline-like список версий: version number, filename, size, date, exists_on_disk badge
- Tap на версию → download конкретной версии
- Action: "Retain latest N" → POST /retain

### 5.7 SettingsScreen (Phase 2)
**Путь:** `/settings`
**Назначение:** конфигурация.

Содержит:
- Current API URL (display + edit)
- Auth mode indicator
- Debug mode toggle (показывает raw JSON responses)
- "Disconnect" button → clear credentials → redirect to ConnectScreen
- App version

### Навигация (screen flow)

```
ConnectScreen ──┬──→ ProjectsScreen ──┬──→ CreateProjectScreen
                │                     ├──→ ProjectStatusScreen ──→ ArtifactScreen
                │                     │                          └──→ VersionsScreen
                │                     └──→ SettingsScreen
                └── (auto-redirect if credentials saved)
```

---

## 6. Data flow

### Request flow (пример: загрузка статуса)

```
StatusScreen widget
  → ref.watch(projectStatusStreamProvider(id))
    → StatusApi.getStatus(id)
      → Dio GET /api/v1/client/status/{id}
        → Headers: { X-API-Key: xxx } or { Authorization: Bearer xxx }
        → Backend → JSON response
      ← StatusDto.fromJson(json)
    ← StatusMapper.toDomain(dto) → ProjectStatus (domain model)
  ← AsyncValue<ProjectStatus>
    ← widget rebuilds: loading → data → error
```

### Три типа моделей

**1. DTO (data layer)** — 1:1 с JSON от backend. Генерируется через `json_serializable`.
```dart
@JsonSerializable()
class ProjectStatusDto {
  final String project_id;
  final String status;
  final String status_label;
  final int progress_percent;
  final bool artifact_available;
  // ... все поля из backend DTO
  factory ProjectStatusDto.fromJson(Map<String, dynamic> json) => ...;
}
```

**2. Domain model (domain layer)** — то, что нужно UI. Типизировано, с enums, computed properties.
```dart
class ProjectStatus {
  final String id;
  final String name;
  final ProjectState state;     // enum: pending, inProgress, completed, failed, archived
  final int progressPercent;
  final bool artifactAvailable;
  final QualityScore? quality;
  final List<Feature> features;
  final TechStack? techStack;
  // computed
  bool get isTerminal => state.isTerminal;
  Duration get pollInterval => state == ProjectState.pending
    ? Duration(seconds: 3) : Duration(seconds: 5);
}
```

**3. Request model (data layer)** — для POST запросов.
```dart
class CreateProjectRequest {
  final String text;
  final String outputFormat;
  final Map<String, dynamic>? metadata;
  Map<String, dynamic> toJson() => ...;
}
```

### Mapper pattern

```dart
class StatusMapper {
  static ProjectStatus toDomain(ProjectStatusDto dto) {
    return ProjectStatus(
      id: dto.project_id,
      name: dto.project_name,
      state: ProjectState.fromString(dto.status),
      progressPercent: dto.progress_percent,
      // ... map all fields, parse dates, convert enums
    );
  }
}
```

---

## 7. Backend endpoint mapping

| Endpoint | Method | Feature | Screen | Описание |
|----------|--------|---------|--------|----------|
| `POST /api/v1/generate` | POST | create_project | CreateProjectScreen | Запуск генерации |
| `GET /api/v1/client/projects` | GET | projects | ProjectsScreen | Список проектов (paginated) |
| `GET /api/v1/client/status/{id}` | GET | project_status | ProjectStatusScreen | Полный статус (flat DTO) |
| `GET /api/v1/client/download/{id}/info` | GET | artifact | ArtifactScreen | Metadata артефакта |
| `GET /api/v1/download/{id}` | GET | artifact | ArtifactScreen | Download binary (zip stream) |
| `GET /api/v1/projects/{id}/transitions` | GET | lifecycle | StatusScreen (AppBar) | Допустимые transitions |
| `POST /api/v1/projects/{id}/archive` | POST | lifecycle | StatusScreen / sheet | Архивация |
| `POST /api/v1/projects/{id}/retry` | POST | lifecycle | StatusScreen / sheet | Retry failed |
| `POST /api/v1/projects/{id}/transition` | POST | lifecycle | LifecycleSheet | Произвольный transition |
| `GET /api/v1/projects/{id}/versions` | GET | lifecycle | VersionsScreen | Список версий |
| `POST /api/v1/projects/{id}/retain` | POST | lifecycle | VersionsScreen | Prune старых версий |
| `GET /api/v1/health` | GET | auth | ConnectScreen | Проверка доступности |

### Auth headers

Все запросы к `/api/v1/client/*` и `/api/v1/generate` передают:
- API Key mode: `X-API-Key: {key}`
- Bearer mode: `Authorization: Bearer {jwt_token}`

Interceptor в Dio добавляет заголовок автоматически на основе сохранённого auth state.

---

## 8. Polling strategy

### Механизм

```
PollingController<T> {
  Stream<T> poll({
    required Future<T> Function() fetcher,
    required bool Function(T) isDone,
    Duration initialInterval = 2s,
    Duration maxInterval = 10s,
    double backoffMultiplier = 1.5,
    int maxRetries = 120,   // ~ 10 min max
  })
}
```

### Интервалы

| Статус | Интервал polling |
|--------|-----------------|
| `pending` | 2 секунды |
| `in_progress`, progress < 50% | 3 секунды |
| `in_progress`, progress >= 50% | 5 секунд |
| `completed` / `failed` / `archived` | STOP |

### Правила

1. Polling запускается только когда StatusScreen видим (Riverpod `autoDispose` отменяет stream при уходе).
2. При сетевой ошибке — exponential backoff до maxInterval, но не останавливать polling.
3. При 404 — остановить polling, показать "Project not found".
4. При 401 — остановить polling, redirect на ConnectScreen.
5. ProjectsScreen НЕ поллит — используется pull-to-refresh.
6. Фоновый polling при возвращении на StatusScreen: stream автоматически перезапускается через Riverpod.

### Оптимизации

- При переходе на StatusScreen с `status=completed` в list data — не запускать polling, сразу показать.
- Cache последнее значение в provider — при навигации назад-вперёд не мигает loading.

---

## 9. Download strategy

### Flow

```
1. User taps "Download" on ArtifactScreen
2. → GET /api/v1/client/download/{id}/info
   ← ArtifactMetadata: size, format, name
3. Check available disk space
4. → GET /api/v1/download/{id}  (stream response)
   ← binary stream with Content-Length
5. Write to app-specific directory:
   - Android: getExternalStorageDirectory() / Downloads
   - iOS: getApplicationDocumentsDirectory()
6. Track progress via Dio onReceiveProgress → StreamNotifier
7. On complete → show "Open" / "Share" buttons
```

### Download state

```dart
sealed class DownloadState {
  Idle()
  Preparing()                          // fetching metadata
  Downloading(double progress, int bytesReceived, int totalBytes)
  Completed(String filePath)
  Failed(String error, bool retryable)
}
```

### Error handling

- Network error mid-download → `Failed(retryable: true)` → show Retry button.
- 404 → artifact was deleted → `Failed(retryable: false)`.
- Disk full → catch IOException → show "Not enough storage".
- Файл уже скачан → check by filename + size → show "Already downloaded. Open?"

### Share

```dart
// После скачивания
await Share.shareXFiles([XFile(filePath)], text: 'Generated project: $name');
```

---

## 10. Error handling strategy

### Central error model

```dart
enum AppErrorType {
  network,         // no connection, timeout
  unauthorized,    // 401 → redirect to ConnectScreen
  forbidden,       // 403 → access denied message
  notFound,        // 404 → "Project not found"
  conflict,        // 409 → invalid lifecycle transition
  serverError,     // 500+ → "Server error, try again"
  generationFailed,// status=failed → show error from backend
  validationError, // client-side or 422 → field-level errors
  unknown,
}

class AppError {
  final AppErrorType type;
  final String userMessage;     // shown to user
  final String? technicalDetail;// for debug mode
  final bool retryable;         // show Retry button?
  final String? errorCode;      // backend code: "project_not_found" etc.
}
```

### Interceptor error mapping (Dio)

```dart
// api_interceptors.dart
class ErrorInterceptor extends Interceptor {
  void onError(DioException err, handler) {
    switch (err.response?.statusCode) {
      case 401: throw UnauthorizedError();     // → router redirect to /connect
      case 404: throw NotFoundError(err.response?.data);
      case 409: throw ConflictError(err.response?.data);
      case 400: throw BadRequestError(err.response?.data);
    }
    if (err.type == DioExceptionType.connectionTimeout) {
      throw NetworkError("Connection timeout");
    }
    throw ServerError(err.message);
  }
}
```

### UI error patterns

| Ситуация | UI |
|----------|-----|
| 401 on any request | SnackBar "Session expired" + auto-redirect to ConnectScreen |
| 404 project | Full-screen "Project not found" + "Back to list" button |
| Network error на ProjectsScreen | Banner сверху "No connection" + retry + показать cached list |
| Network error на StatusScreen | Banner "Offline" + показать last known state + auto-retry |
| Generation failed | Red error card в StatusScreen с backend error message |
| Invalid transition (409) | SnackBar "Cannot archive: project is still generating" |
| Download failed | SnackBar + Retry button прямо на ArtifactScreen |
| Empty project list | Friendly empty state illustration + "Create first project" CTA |

### Backend error contract integration

Backend возвращает:
```json
{"code": "project_not_found", "message": "...", "retryable": false}
```

DTO:
```dart
class ApiErrorDto {
  final String code;
  final String message;
  final bool retryable;
}
```

Flutter маппит `code` → `AppErrorType`, `message` → `userMessage`, `retryable` → кнопка Retry.

---

## 11. MVP Roadmap

### Phase 1: Minimal Working Client (1-2 недели)

**Цель:** подключиться к backend, создать проект, увидеть результат, скачать.

| # | Задача | Файлы |
|---|--------|-------|
| 1 | Project setup: Flutter create, add deps (riverpod, dio, go_router, flutter_secure_storage, json_serializable) | pubspec.yaml, analysis_options |
| 2 | Core API client: Dio + auth interceptor + error interceptor | core/api/* |
| 3 | Auth: ConnectScreen, save API key, secure storage | features/auth/* |
| 4 | Projects list: fetch + display | features/projects/* |
| 5 | Create project: form + POST /generate + navigate to status | features/create_project/* |
| 6 | Project status: fetch + display + polling | features/project_status/* |
| 7 | Download: artifact info + binary download + save | features/artifact/* (basic) |

**Результат Phase 1:** можно подключиться, создать проект, наблюдать генерацию, скачать результат.

### Phase 2: Stable Useful Client (1-2 недели)

**Цель:** production-ready UX, error handling, lifecycle.

| # | Задача |
|---|--------|
| 1 | Error handling: global interceptor, error widgets, 401 auto-redirect |
| 2 | Offline resilience: show cached data, retry banners |
| 3 | Pull-to-refresh на ProjectsScreen |
| 4 | Lifecycle: archive, retry (bottom sheet на StatusScreen) |
| 5 | Allowed transitions: fetch + show only valid actions |
| 6 | Settings screen: URL, auth, disconnect |
| 7 | Loading states: shimmer, skeleton screens |
| 8 | Empty states: illustrations |
| 9 | Polish StatusScreen: quality section, pipeline section |

**Результат Phase 2:** приложение можно показать пользователям, оно не падает, обрабатывает ошибки, lifecycle работает.

### Phase 3: Advanced UX (2+ недели)

**Цель:** полноценный клиент.

| # | Задача |
|---|--------|
| 1 | Versions screen: список версий, download конкретной, retain |
| 2 | Bearer/JWT token support: decode, show expiry, auto-refresh |
| 3 | Pagination: infinite scroll на ProjectsScreen |
| 4 | Search/filter: по статусу, дате, features |
| 5 | Share artifact: system share sheet |
| 6 | Push notifications (или foreground service): generation complete |
| 7 | Dark theme |
| 8 | Localization (l10n) |
| 9 | Deep links: open project by URL |
| 10 | Tablet layout: master-detail на ProjectsScreen |

---

## 12. Зависимости (pubspec.yaml)

```yaml
dependencies:
  flutter:
    sdk: flutter

  # State management
  flutter_riverpod: ^2.5.0
  riverpod_annotation: ^2.3.0

  # Networking
  dio: ^5.4.0

  # Routing
  go_router: ^14.0.0

  # Storage
  flutter_secure_storage: ^9.0.0
  shared_preferences: ^2.2.0

  # Serialization
  json_annotation: ^4.9.0
  freezed_annotation: ^2.4.0

  # UI
  shimmer: ^3.0.0
  share_plus: ^9.0.0
  path_provider: ^2.1.0
  intl: ^0.19.0
  url_launcher: ^6.2.0

dev_dependencies:
  build_runner: ^2.4.0
  json_serializable: ^6.8.0
  freezed: ^2.5.0
  riverpod_generator: ^2.4.0
  riverpod_lint: ^2.3.0
  flutter_test:
    sdk: flutter
  mocktail: ^1.0.0
```

---

## 13. Ключевые архитектурные решения

**GoRouter** для навигации — поддерживает guards (redirect на /connect если нет auth), deep links, nested navigation.

**Freezed + json_serializable** для DTO и domain моделей — immutable classes, union types для sealed states, генерация fromJson/toJson.

**flutter_secure_storage** для credentials — не SharedPreferences, а encrypted storage.

**Dio** вместо http — interceptors, download progress, cancel tokens для polling, connection timeout.

**autoDispose** на всех providers по умолчанию — при уходе с экрана polling останавливается, memory освобождается.

**Mapper pattern** (не автоматический) — backend DTOs маппятся в domain models руками, потому что naming conventions отличаются (snake_case → camelCase), нужны enum conversions, computed fields.

---

## 14. Что НЕ делать

1. Не использовать GetX — плохо тестируется, неявные зависимости.
2. Не хранить API key в SharedPreferences — только secure storage.
3. Не делать polling через Timer — использовать Stream + autoDispose.
4. Не дублировать бизнес-логику backend на клиенте — backend уже считает progress_percent, status_label, artifact_available. Flutter просто отображает.
5. Не создавать God-provider — один provider на один concern.
6. Не миксовать data и domain — DTO не должен утекать в widgets.
