import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../core/error/error_handler.dart';
import '../../../shared/constants/strings.dart';
import '../../../shared/widgets/nk_app_bar.dart';
import '../../../shared/widgets/nk_empty_state.dart';
import '../../../shared/widgets/nk_error_view.dart';
import '../../../shared/widgets/nk_loading.dart';
import '../projects_providers.dart';
import 'project_card.dart';

class ProjectsScreen extends ConsumerWidget {
  const ProjectsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final projectsAsync = ref.watch(projectListProvider);

    return Scaffold(
      appBar: NKAppBar(
        title: Strings.projects,
        actions: [
          IconButton(
            icon: const Icon(Icons.code_outlined),
            tooltip: 'Code Assistant',
            onPressed: () => context.push('/code'),
          ),
          IconButton(
            icon: const Icon(Icons.psychology_outlined),
            tooltip: 'Ask NK',
            onPressed: () => context.push('/ask'),
          ),
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            onPressed: () => context.push('/settings'),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => context.push('/projects/create'),
        child: const Icon(Icons.add),
      ),
      body: projectsAsync.when(
        loading: () => const NKShimmerList(),
        error: (err, _) => FullScreenError(
          error: handleError(err),
          onRetry: () => ref.read(projectListProvider.notifier).refresh(),
        ),
        data: (projects) {
          if (projects.isEmpty) {
            return NKEmptyState(
              icon: Icons.folder_open_rounded,
              title: Strings.noProjects,
              subtitle: Strings.noProjectsSub,
              actionLabel: Strings.createProject,
              onAction: () => context.push('/projects/create'),
            );
          }

          return RefreshIndicator(
            onRefresh: () =>
                ref.read(projectListProvider.notifier).refresh(),
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(vertical: 8),
              itemCount: projects.length,
              itemBuilder: (_, i) => ProjectCard(
                project: projects[i],
                onTap: () => context.push('/projects/${projects[i].id}'),
              ),
            ),
          );
        },
      ),
    );
  }
}
