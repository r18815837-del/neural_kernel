# NK Chat — Flutter desktop клиент для Ollama

Локальный ChatGPT-подобный интерфейс: чат, markdown, подсветка кода, стриминг
ответов, выбор модели, история чатов. Работает **полностью локально**, ходит
в твою Ollama на `localhost:11434`.

## Что умеет

- **Чат со стримингом** — ответ появляется токен за токеном.
- **Markdown + подсветка кода** — code blocks с языками (Python, JS и т.д.).
- **Сайдбар** — список всех чатов, переключение, удаление.
- **Выбор модели на лету** — все твои `ollama pull` сразу видны.
- **System prompt** и **temperature** — через иконку настроек.
- **Stop-кнопка** — прервать генерацию.
- **Copy** — на каждом ответе и каждом code-блоке.
- **Тёмная тема.**

## Требования

- Flutter SDK 3.16+
- Android Studio (или VS Code)
- Visual Studio 2022 Build Tools с компонентом **Desktop development with C++**
- Ollama запущена, модели скачаны (у тебя уже есть `qwen2.5-coder:7b`)

Проверь: `flutter doctor` — должно быть зелёное для Windows и Visual Studio.

---

## Первый запуск (в Android Studio)

### 1. Сгенерировать платформенные файлы для Windows

Flutter-проект без папки `windows/` не скомпилируется под desktop. Сгенерируй
её поверх моих исходников:

```powershell
cd C:\Users\r1881\Downloads\neural_kernel\nk_chat_app
flutter create . --platforms=windows --project-name=nk_chat --org=com.nk
```

**Важно:** если `flutter create` спросит, перезаписывать ли существующие файлы
(`pubspec.yaml`, `lib/main.dart`, `lib/widgets.dart` и т.д.) — **отвечай `n`**
(no). Нам нужны мои версии. Он просто добавит отсутствующие `windows/`,
`.metadata`, `.gitignore` и т.п.

### 2. Установить зависимости

```powershell
flutter pub get
```

### 3. Включить Windows-desktop (если ещё не)

```powershell
flutter config --enable-windows-desktop
```

### 4. Открыть в Android Studio

1. **File → Open** → выбери папку `C:\Users\r1881\Downloads\neural_kernel\nk_chat_app`.
2. Дождись индексации (~30 сек — в правом нижнем углу будет крутиться прогресс).
3. В правом верхнем углу есть панель с выбором таргета (обычно между `main.dart`
   и зелёной ▶). В дропдауне **Device selector** выбери **Windows (desktop)**.
4. Нажми зелёный **▶ Run** (или `Shift+F10`).

### 5. Первая сборка

Первая компиляция долгая — Flutter тянет и собирает Windows shell. Займёт
**2–5 минут**. Последующие сборки — секунды.

Когда откроется окно приложения:

- В левом нижнем углу должно быть **зелёный кружок + «Ollama: N models»**.
- В заголовке — селектор моделей. Если пусто — кликни иконку ↻ в левом
  нижнем углу.
- Нажми **New chat**, напиши вопрос — ответ пойдёт стримом.

---

## Повседневный запуск

После первой настройки:

```powershell
cd C:\Users\r1881\Downloads\neural_kernel\nk_chat_app
flutter run -d windows
```

Или в Android Studio просто ▶ Run.

---

## Горячие клавиши

| Что | Как |
|-----|-----|
| Отправить | `Enter` |
| Новая строка | `Shift+Enter` |
| Новый чат | Клик **New chat** в сайдбаре |
| Настройки чата | Иконка шестерёнки в правом верхнем углу |
| Stop генерацию | Красная кнопка **Stop** во время ответа |

---

## Структура кода

```
nk_chat_app/
├── pubspec.yaml
├── README.md
├── analysis_options.yaml
├── lib/
│   ├── main.dart              ← точка входа + тёмная тема
│   ├── models.dart            ← Message, Conversation
│   ├── ollama_service.dart    ← HTTP-клиент к Ollama (стриминг)
│   ├── chat_controller.dart   ← state (ChangeNotifier)
│   ├── chat_screen.dart       ← главный экран
│   └── widgets.dart           ← MessageBubble, Sidebar, ModelSelector, Settings
└── windows/                   ← генерируется flutter create
```

Всё помещается в ~1400 строк Dart. Зависимостей минимум: `http`,
`flutter_markdown`, `flutter_highlight`, `provider`, `uuid`.

---

## Troubleshooting

### «Ollama offline» в нижнем углу

Ollama не отвечает. Проверь:

1. Иконка ламы в трее есть?
2. `curl http://localhost:11434/api/tags` — возвращает JSON?

### Нет моделей в селекторе

```powershell
ollama pull qwen2.5-coder:7b
```

Потом в приложении → иконка ↻ слева снизу.

### Ошибка «CMake not found» при сборке

Нужны Visual Studio 2022 Build Tools с workload **Desktop development with C++**.
Скачай [Visual Studio Community](https://visualstudio.microsoft.com/downloads/) →
при установке выбери этот workload.

### «Windows» не появляется в Device Selector

```powershell
flutter config --enable-windows-desktop
flutter doctor
```

После — перезапусти Android Studio.

### Долгая первая сборка

Нормально. ~200 МБ зависимостей собирается впервые. Следующие сборки — 5–30 сек.

### Markdown/code выглядит странно

Убедись, что модель возвращает настоящие ```code fences```. Если нет —
скорректируй system prompt в настройках чата.

---

## Что добавить дальше

1. **Персистентность** — сохранять чаты в SQLite / файл, чтобы не терялись при закрытии.
2. **RAG** — запрашивать Qdrant (уже поднят в `rag/`) перед отправкой в LLM.
3. **File attachments** — drag & drop файлов в чат.
4. **Regenerate** — перезапрос последнего ответа.
5. **Export** — сохранить чат в markdown.

Все расширения — маленькие довески поверх `ChatController`. Помогу, когда захочешь.
