import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../core/auth/auth_models.dart';
import '../../../core/auth/auth_provider.dart';
import 'connect_form.dart';

class ConnectScreen extends ConsumerStatefulWidget {
  const ConnectScreen({super.key});

  @override
  ConsumerState<ConnectScreen> createState() => _ConnectScreenState();
}

class _ConnectScreenState extends ConsumerState<ConnectScreen> {
  final _urlController = TextEditingController();
  final _secretController = TextEditingController();
  AuthMode _mode = AuthMode.none;
  bool _loading = false;
  String? _error;

  @override
  void dispose() {
    _urlController.dispose();
    _secretController.dispose();
    super.dispose();
  }

  Future<void> _connect() async {
    final url = _urlController.text.trim();
    final secret = _secretController.text.trim();

    if (url.isEmpty) {
      setState(() => _error = 'Please enter the server URL');
      return;
    }
    if (_mode != AuthMode.none && secret.isEmpty) {
      setState(() => _error = 'Please enter your credentials');
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
    });

    final creds = AuthCredentials(
      baseUrl: url,
      mode: _mode,
      secret: secret,
    );

    final service = ref.read(authServiceProvider);
    final validationError = await service.validate(creds);

    if (!mounted) return;

    if (validationError != null) {
      setState(() {
        _loading = false;
        _error = validationError;
      });
      return;
    }

    // Success — persist and update state.
    await ref.read(authStateProvider.notifier).login(creds);

    if (mounted) {
      context.go('/projects');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(32),
            child: ConnectForm(
              urlController: _urlController,
              secretController: _secretController,
              mode: _mode,
              loading: _loading,
              error: _error,
              onModeChanged: (m) => setState(() => _mode = m),
              onConnect: _connect,
            ),
          ),
        ),
      ),
    );
  }
}
