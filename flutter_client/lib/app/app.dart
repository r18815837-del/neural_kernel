import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../shared/constants/strings.dart';
import 'router.dart';
import 'theme.dart';

/// Root application widget.
class NKApp extends ConsumerWidget {
  const NKApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final router = ref.watch(routerProvider);

    return MaterialApp.router(
      title: Strings.appName,
      theme: NKTheme.light(),
      routerConfig: router,
      debugShowCheckedModeBanner: false,
    );
  }
}
