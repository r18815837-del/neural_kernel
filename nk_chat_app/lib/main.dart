import 'dart:io' show Platform;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_acrylic/flutter_acrylic.dart';
import 'package:provider/provider.dart';
import 'package:window_manager/window_manager.dart';

import 'chat_controller.dart';
import 'chat_screen.dart';

/// Шрифтовой стек: Segoe UI Variable (Windows 11) → Segoe UI → system.
const kUiFontFamily = 'Segoe UI Variable';
const kUiFontFamilyFallback = <String>['Segoe UI', 'Arial'];

/// Моноширинный стек: Cascadia Mono → Consolas → monospace.
const kMonoFontFamily = 'Cascadia Mono';
const kMonoFontFamilyFallback = <String>['Consolas', 'monospace'];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Windows 11 Mica / Acrylic blur.
  if (Platform.isWindows) {
    try {
      await Window.initialize();
      await Window.setEffect(
        effect: WindowEffect.acrylic,
        color: const Color(0xCC1E1F22),
        dark: true,
      );
    } catch (_) {}
  }

  // Custom title bar через window_manager.
  try {
    await windowManager.ensureInitialized();
    const windowOptions = WindowOptions(
      size: Size(1280, 820),
      minimumSize: Size(800, 500),
      center: true,
      backgroundColor: Colors.transparent,
      skipTaskbar: false,
      titleBarStyle: TitleBarStyle.hidden,
      title: 'NK Chat',
    );
    await windowManager.waitUntilReadyToShow(windowOptions, () async {
      await windowManager.show();
      await windowManager.focus();
    });
  } catch (_) {
    // Если window_manager недоступен — просто стандартное окно.
  }

  final controller = ChatController();
  await controller.init();

  runApp(NkChatApp(controller: controller));
}

class NkChatApp extends StatelessWidget {
  const NkChatApp({super.key, required this.controller});

  final ChatController controller;

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider.value(
      value: controller,
      child: MaterialApp(
        title: 'NK Chat',
        debugShowCheckedModeBanner: false,
        theme: _buildTheme(),
        home: const _ShortcutsWrapper(child: _WindowFrame(child: ChatScreen())),
      ),
    );
  }
}

/// Обёртка, добавляющая кастомный title bar сверху.
class _WindowFrame extends StatelessWidget {
  const _WindowFrame({required this.child});
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const _TitleBar(),
        Expanded(child: child),
      ],
    );
  }
}

class _TitleBar extends StatefulWidget {
  const _TitleBar();

  @override
  State<_TitleBar> createState() => _TitleBarState();
}

class _TitleBarState extends State<_TitleBar> with WindowListener {
  bool _maximized = false;

  @override
  void initState() {
    super.initState();
    windowManager.addListener(this);
    _checkMaximized();
  }

  @override
  void dispose() {
    windowManager.removeListener(this);
    super.dispose();
  }

  Future<void> _checkMaximized() async {
    try {
      final m = await windowManager.isMaximized();
      if (mounted) setState(() => _maximized = m);
    } catch (_) {}
  }

  @override
  void onWindowMaximize() => _checkMaximized();

  @override
  void onWindowUnmaximize() => _checkMaximized();

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 32,
      color: const Color(0xE817181A),
      child: Row(
        children: [
          // Бренд.
          const SizedBox(width: 12),
          Container(
            width: 14,
            height: 14,
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [Color(0xFF7C9CFF), Color(0xFF6FE0C5)],
              ),
              borderRadius: BorderRadius.circular(3),
            ),
          ),
          const SizedBox(width: 8),
          const Text(
            'NK Chat',
            style: TextStyle(
              color: Colors.white70,
              fontSize: 11,
              fontWeight: FontWeight.w500,
            ),
          ),
          // Draggable area — всё остальное до кнопок.
          Expanded(
            child: GestureDetector(
              behavior: HitTestBehavior.translucent,
              onPanStart: (_) {
                try {
                  windowManager.startDragging();
                } catch (_) {}
              },
              onDoubleTap: () async {
                try {
                  if (await windowManager.isMaximized()) {
                    await windowManager.unmaximize();
                  } else {
                    await windowManager.maximize();
                  }
                } catch (_) {}
              },
              child: const SizedBox.expand(),
            ),
          ),
          _WindowButton(
            icon: Icons.minimize,
            onTap: () {
              try {
                windowManager.minimize();
              } catch (_) {}
            },
          ),
          _WindowButton(
            icon: _maximized
                ? Icons.filter_none_outlined
                : Icons.crop_square,
            iconSize: _maximized ? 12 : 14,
            onTap: () async {
              try {
                if (_maximized) {
                  await windowManager.unmaximize();
                } else {
                  await windowManager.maximize();
                }
              } catch (_) {}
            },
          ),
          _WindowButton(
            icon: Icons.close,
            hoverColor: const Color(0xFFE81123),
            onTap: () {
              try {
                windowManager.close();
              } catch (_) {}
            },
          ),
        ],
      ),
    );
  }
}

class _WindowButton extends StatefulWidget {
  const _WindowButton({
    required this.icon,
    required this.onTap,
    this.iconSize = 14,
    this.hoverColor = const Color(0xFF2A2B2E),
  });

  final IconData icon;
  final VoidCallback onTap;
  final double iconSize;
  final Color hoverColor;

  @override
  State<_WindowButton> createState() => _WindowButtonState();
}

class _WindowButtonState extends State<_WindowButton> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: Container(
          width: 46,
          height: 32,
          color: _hovered ? widget.hoverColor : Colors.transparent,
          alignment: Alignment.center,
          child: Icon(
            widget.icon,
            size: widget.iconSize,
            color: _hovered && widget.hoverColor.red > 200
                ? Colors.white
                : Colors.white70,
          ),
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════════
// Shortcuts
// ════════════════════════════════════════════════════════════════════════
class _ShortcutsWrapper extends StatelessWidget {
  const _ShortcutsWrapper({required this.child});
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Shortcuts(
      shortcuts: const {
        SingleActivator(LogicalKeyboardKey.keyN, control: true):
            _NewChatIntent(),
        SingleActivator(LogicalKeyboardKey.keyK, control: true):
            _FocusSearchIntent(),
        SingleActivator(LogicalKeyboardKey.keyL, control: true):
            _FocusInputIntent(),
        SingleActivator(LogicalKeyboardKey.keyF, control: true):
            _FindInChatIntent(),
        SingleActivator(LogicalKeyboardKey.keyC, control: true, shift: true):
            _CopyLastAssistantIntent(),
        SingleActivator(LogicalKeyboardKey.keyE, control: true, shift: true):
            _ExportIntent(),
        SingleActivator(LogicalKeyboardKey.escape): _CancelIntent(),
      },
      child: Actions(
        actions: {
          _NewChatIntent: CallbackAction<_NewChatIntent>(
            onInvoke: (_) {
              context.read<ChatController>().createConversation();
              return null;
            },
          ),
          _FocusSearchIntent: CallbackAction<_FocusSearchIntent>(
            onInvoke: (_) {
              SearchFocus.requestFocus();
              return null;
            },
          ),
          _FocusInputIntent: CallbackAction<_FocusInputIntent>(
            onInvoke: (_) {
              InputFocus.requestFocus();
              return null;
            },
          ),
          _FindInChatIntent: CallbackAction<_FindInChatIntent>(
            onInvoke: (_) {
              InChatSearchBus.open();
              return null;
            },
          ),
          _CancelIntent: CallbackAction<_CancelIntent>(
            onInvoke: (_) {
              InChatSearchBus.close();
              return null;
            },
          ),
          _CopyLastAssistantIntent: CallbackAction<_CopyLastAssistantIntent>(
            onInvoke: (_) {
              final ctrl = context.read<ChatController>();
              final conv = ctrl.activeConversation;
              if (conv == null || conv.messages.isEmpty) return null;
              final last = conv.messages.reversed.firstWhere(
                (m) => m.role.name == 'assistant',
                orElse: () => conv.messages.last,
              );
              Clipboard.setData(ClipboardData(text: last.content));
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Last assistant message copied'),
                  duration: Duration(seconds: 1),
                  behavior: SnackBarBehavior.floating,
                ),
              );
              return null;
            },
          ),
          _ExportIntent: CallbackAction<_ExportIntent>(
            onInvoke: (_) {
              final ctrl = context.read<ChatController>();
              final md = ctrl.exportActiveAsMarkdown();
              if (md.isEmpty) return null;
              Clipboard.setData(ClipboardData(text: md));
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Chat exported as markdown → clipboard'),
                  duration: Duration(seconds: 2),
                  behavior: SnackBarBehavior.floating,
                ),
              );
              return null;
            },
          ),
        },
        child: Focus(autofocus: true, child: child),
      ),
    );
  }
}

class _NewChatIntent extends Intent {
  const _NewChatIntent();
}

class _FocusSearchIntent extends Intent {
  const _FocusSearchIntent();
}

class _FocusInputIntent extends Intent {
  const _FocusInputIntent();
}

class _FindInChatIntent extends Intent {
  const _FindInChatIntent();
}

class _CancelIntent extends Intent {
  const _CancelIntent();
}

class _CopyLastAssistantIntent extends Intent {
  const _CopyLastAssistantIntent();
}

class _ExportIntent extends Intent {
  const _ExportIntent();
}

/// Focus-узлы, к которым обращаются шорткаты.
class SearchFocus {
  static FocusNode? node;
  static void requestFocus() => node?.requestFocus();
}

class InputFocus {
  static FocusNode? node;
  static void requestFocus() => node?.requestFocus();
}

/// Простенький event-bus для управления in-chat search баром.
class InChatSearchBus {
  static final List<void Function()> _openListeners = [];
  static final List<void Function()> _closeListeners = [];

  static void onOpen(void Function() listener) => _openListeners.add(listener);
  static void onClose(void Function() listener) =>
      _closeListeners.add(listener);
  static void offOpen(void Function() listener) =>
      _openListeners.remove(listener);
  static void offClose(void Function() listener) =>
      _closeListeners.remove(listener);

  static void open() {
    for (final l in _openListeners) l();
  }

  static void close() {
    for (final l in _closeListeners) l();
  }
}

ThemeData _buildTheme() {
  const bg = Color(0xE81E1F22);
  const accent = Color(0xFF7C9CFF);
  const text = Color(0xFFE6E7E9);

  final base = TextStyle(
    fontFamily: kUiFontFamily,
    fontFamilyFallback: kUiFontFamilyFallback,
    color: text,
    height: 1.5,
  );

  return ThemeData(
    brightness: Brightness.dark,
    useMaterial3: true,
    fontFamily: kUiFontFamily,
    fontFamilyFallback: kUiFontFamilyFallback,
    colorScheme: const ColorScheme.dark(
      primary: accent,
      secondary: accent,
      surface: bg,
      onSurface: text,
    ),
    scaffoldBackgroundColor: bg,
    canvasColor: const Color(0xE817181A),
    cardColor: const Color(0xFF2A2B2E),
    dividerColor: Colors.white12,
    textTheme: TextTheme(
      bodySmall: base.copyWith(fontSize: 12, height: 1.45),
      bodyMedium: base.copyWith(fontSize: 14, height: 1.55),
      bodyLarge: base.copyWith(fontSize: 15, height: 1.6),
      titleSmall: base.copyWith(fontSize: 13, fontWeight: FontWeight.w600),
      titleMedium: base.copyWith(fontSize: 14, fontWeight: FontWeight.w600),
      labelLarge: base.copyWith(fontSize: 13, fontWeight: FontWeight.w500),
    ),
    inputDecorationTheme: const InputDecorationTheme(
      filled: true,
      fillColor: Color(0xFF26272A),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.all(Radius.circular(10)),
        borderSide: BorderSide.none,
      ),
      hintStyle: TextStyle(color: Colors.white38),
    ),
    iconTheme: const IconThemeData(color: Colors.white70),
    scrollbarTheme: ScrollbarThemeData(
      thumbVisibility: MaterialStateProperty.all(false),
      thickness: MaterialStateProperty.resolveWith<double>((states) {
        if (states.contains(MaterialState.hovered) ||
            states.contains(MaterialState.dragged)) {
          return 8;
        }
        return 6;
      }),
      thumbColor: MaterialStateProperty.resolveWith<Color>((states) {
        if (states.contains(MaterialState.dragged)) return Colors.white38;
        if (states.contains(MaterialState.hovered)) return Colors.white30;
        return Colors.white12;
      }),
      radius: const Radius.circular(3),
      crossAxisMargin: 2,
    ),
  );
}
