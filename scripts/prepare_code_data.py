"""Prepare code corpus for training the Neural Kernel coding assistant.

Downloads Python and Dart code snippets from public sources:
  - Python: stdlib docs examples, common patterns, algorithms
  - Dart: Flutter patterns, common widgets, async patterns

Usage:
    python scripts/prepare_code_data.py
    python scripts/prepare_code_data.py --lang python --max-snippets 5000
    python scripts/prepare_code_data.py --lang all
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data/corpus")
_USER_AGENT = "NeuralKernel/1.0"


# ------------------------------------------------------------------
# Built-in code patterns (always available, no network needed)
# ------------------------------------------------------------------

PYTHON_PATTERNS = [
    # --- Data Structures ---
    '''# Python: List comprehension with filtering
def get_even_squares(numbers):
    """Return squares of even numbers."""
    return [x ** 2 for x in numbers if x % 2 == 0]

# Example: get_even_squares([1, 2, 3, 4, 5]) -> [4, 16]''',

    '''# Python: Dictionary comprehension
def invert_dict(d):
    """Swap keys and values in a dictionary."""
    return {v: k for k, v in d.items()}

# Example: invert_dict({"a": 1, "b": 2}) -> {1: "a", 2: "b"}''',

    '''# Python: Merge two sorted lists
def merge_sorted(a, b):
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result''',

    '''# Python: Binary search
def binary_search(arr, target):
    """Find target in sorted array. Return index or -1."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1''',

    '''# Python: Flatten nested list
def flatten(lst):
    """Recursively flatten a nested list."""
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

# Example: flatten([1, [2, [3, 4]], 5]) -> [1, 2, 3, 4, 5]''',

    # --- Classes and OOP ---
    '''# Python: Dataclass with validation
from dataclasses import dataclass, field

@dataclass
class User:
    name: str
    email: str
    age: int = 0
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.age < 0:
            raise ValueError(f"Age must be >= 0, got {self.age}")
        if "@" not in self.email:
            raise ValueError(f"Invalid email: {self.email}")''',

    '''# Python: Context manager
class Timer:
    """Context manager that measures execution time."""
    def __init__(self, label=""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        import time
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            print(f"{self.label}: {self.elapsed:.3f}s")

# Usage: with Timer("sort") as t: sorted(big_list)''',

    '''# Python: Iterator protocol
class Range:
    """Custom range iterator."""
    def __init__(self, start, stop, step=1):
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        self.current = self.start
        return self

    def __next__(self):
        if self.current >= self.stop:
            raise StopIteration
        value = self.current
        self.current += self.step
        return value''',

    '''# Python: Decorator with arguments
import functools

def retry(max_attempts=3, delay=1.0):
    """Retry a function on exception."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

# Usage: @retry(max_attempts=5, delay=0.5)''',

    '''# Python: Abstract base class
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        import math
        return math.pi * self.radius ** 2

    def perimeter(self) -> float:
        import math
        return 2 * math.pi * self.radius''',

    # --- Async ---
    '''# Python: Async HTTP requests
import asyncio
import aiohttp

async def fetch_url(session, url):
    """Fetch a single URL."""
    async with session.get(url) as response:
        return await response.text()

async def fetch_all(urls):
    """Fetch multiple URLs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)''',

    '''# Python: Async generator
import asyncio

async def async_range(start, stop):
    """Async generator that yields numbers with delay."""
    for i in range(start, stop):
        await asyncio.sleep(0.1)
        yield i

async def main():
    async for num in async_range(0, 10):
        print(num)''',

    # --- File I/O ---
    '''# Python: Read and process CSV
import csv

def read_csv(path):
    """Read CSV file into list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def write_csv(path, data, fieldnames=None):
    """Write list of dicts to CSV file."""
    if not data:
        return
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)''',

    '''# Python: JSON config loader
import json
from pathlib import Path

def load_config(path="config.json", defaults=None):
    """Load JSON config with defaults."""
    config = defaults or {}
    p = Path(path)
    if p.exists():
        with open(p, "r") as f:
            config.update(json.load(f))
    return config

def save_config(config, path="config.json"):
    """Save config to JSON."""
    with open(path, "w") as f:
        json.dump(config, f, indent=2)''',

    # --- Error handling ---
    '''# Python: Custom exception hierarchy
class AppError(Exception):
    """Base application error."""
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code

class NotFoundError(AppError):
    """Resource not found."""
    def __init__(self, resource, id):
        super().__init__(f"{resource} with id={id} not found", code=404)

class ValidationError(AppError):
    """Input validation failed."""
    def __init__(self, field, message):
        super().__init__(f"Validation error on '{field}': {message}", code=400)
        self.field = field''',

    # --- Testing ---
    '''# Python: Unit test with pytest
import pytest

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def test_divide_normal():
    assert divide(10, 2) == 5.0

def test_divide_negative():
    assert divide(-10, 2) == -5.0

def test_divide_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10, 0)

@pytest.mark.parametrize("a,b,expected", [
    (10, 2, 5.0),
    (0, 5, 0.0),
    (7, 3, 7/3),
])
def test_divide_parametrize(a, b, expected):
    assert divide(a, b) == pytest.approx(expected)''',

    # --- Algorithms ---
    '''# Python: Quick sort
def quicksort(arr):
    """In-place quicksort."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)''',

    '''# Python: BFS graph traversal
from collections import deque

def bfs(graph, start):
    """Breadth-first search. Returns visited nodes in order."""
    visited = set()
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
    return order''',

    '''# Python: LRU Cache implementation
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)''',

    # --- Web / API ---
    '''# Python: FastAPI endpoint
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    quantity: int = 1

items_db: dict[int, Item] = {}
next_id = 1

@app.post("/items", status_code=201)
def create_item(item: Item):
    global next_id
    items_db[next_id] = item
    result = {"id": next_id, **item.model_dump()}
    next_id += 1
    return result

@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"id": item_id, **items_db[item_id].model_dump()}''',

    # --- NumPy / Data ---
    '''# Python: NumPy array operations
import numpy as np

def normalize(arr):
    """Normalize array to [0, 1] range."""
    min_val = arr.min()
    max_val = arr.max()
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def moving_average(arr, window=3):
    """Compute moving average."""
    return np.convolve(arr, np.ones(window) / window, mode="valid")

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))''',
]


DART_PATTERNS = [
    # --- Basics ---
    '''// Dart: Class with named parameters
class User {
  final String name;
  final String email;
  final int age;

  const User({required this.name, required this.email, this.age = 0});

  User copyWith({String? name, String? email, int? age}) {
    return User(
      name: name ?? this.name,
      email: email ?? this.email,
      age: age ?? this.age,
    );
  }

  @override
  String toString() => 'User(name: $name, email: $email, age: $age)';
}''',

    '''// Dart: Extension methods
extension StringExtension on String {
  String capitalize() {
    if (isEmpty) return this;
    return '${this[0].toUpperCase()}${substring(1)}';
  }

  bool get isEmail => RegExp(r'^[\\w.-]+@[\\w.-]+\\.\\w+$').hasMatch(this);

  String truncate(int maxLength, {String suffix = '...'}) {
    if (length <= maxLength) return this;
    return '${substring(0, maxLength - suffix.length)}$suffix';
  }
}''',

    '''// Dart: Sealed class pattern (Dart 3)
sealed class Result<T> {
  const Result();
}

class Success<T> extends Result<T> {
  final T data;
  const Success(this.data);
}

class Failure<T> extends Result<T> {
  final String error;
  const Failure(this.error);
}

// Usage with pattern matching:
// switch (result) {
//   case Success(data: final d): print('Got: $d');
//   case Failure(error: final e): print('Error: $e');
// }''',

    '''// Dart: Async/await with error handling
Future<String> fetchData(String url) async {
  try {
    final response = await http.get(Uri.parse(url));
    if (response.statusCode == 200) {
      return response.body;
    }
    throw HttpException('Failed: ${response.statusCode}');
  } on SocketException {
    throw HttpException('No internet connection');
  } on TimeoutException {
    throw HttpException('Request timed out');
  }
}''',

    '''// Dart: Stream processing
Stream<int> countStream(int max) async* {
  for (int i = 0; i < max; i++) {
    await Future.delayed(Duration(milliseconds: 100));
    yield i;
  }
}

Future<int> sumStream(Stream<int> stream) async {
  int sum = 0;
  await for (final value in stream) {
    sum += value;
  }
  return sum;
}''',

    # --- Flutter ---
    '''// Flutter: Stateful widget with animation
class PulseButton extends StatefulWidget {
  final String label;
  final VoidCallback onPressed;

  const PulseButton({super.key, required this.label, required this.onPressed});

  @override
  State<PulseButton> createState() => _PulseButtonState();
}

class _PulseButtonState extends State<PulseButton>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scale;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    )..repeat(reverse: true);
    _scale = Tween(begin: 1.0, end: 1.1).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ScaleTransition(
      scale: _scale,
      child: ElevatedButton(
        onPressed: widget.onPressed,
        child: Text(widget.label),
      ),
    );
  }
}''',

    '''// Flutter: Riverpod async provider
import 'package:flutter_riverpod/flutter_riverpod.dart';

final userProvider = FutureProvider.autoDispose<User>((ref) async {
  final api = ref.read(apiProvider);
  return await api.getCurrentUser();
});

class UserScreen extends ConsumerWidget {
  const UserScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final userAsync = ref.watch(userProvider);

    return userAsync.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (err, stack) => Center(child: Text('Error: $err')),
      data: (user) => Center(child: Text('Hello, ${user.name}!')),
    );
  }
}''',

    '''// Flutter: Custom painter
class CircleProgressPainter extends CustomPainter {
  final double progress;
  final Color color;

  CircleProgressPainter({required this.progress, this.color = Colors.blue});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2 - 4;

    // Background circle
    final bgPaint = Paint()
      ..color = color.withOpacity(0.2)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4;
    canvas.drawCircle(center, radius, bgPaint);

    // Progress arc
    final fgPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      -pi / 2,
      2 * pi * progress,
      false,
      fgPaint,
    );
  }

  @override
  bool shouldRepaint(CircleProgressPainter old) => old.progress != progress;
}''',

    '''// Flutter: Responsive layout builder
class ResponsiveLayout extends StatelessWidget {
  final Widget mobile;
  final Widget? tablet;
  final Widget? desktop;

  const ResponsiveLayout({
    super.key,
    required this.mobile,
    this.tablet,
    this.desktop,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        if (constraints.maxWidth >= 1024 && desktop != null) {
          return desktop!;
        }
        if (constraints.maxWidth >= 600 && tablet != null) {
          return tablet!;
        }
        return mobile;
      },
    );
  }
}''',

    '''// Flutter: GoRouter configuration
import 'package:go_router/go_router.dart';

final router = GoRouter(
  initialLocation: '/',
  routes: [
    GoRoute(
      path: '/',
      builder: (context, state) => const HomeScreen(),
    ),
    GoRoute(
      path: '/profile/:userId',
      builder: (context, state) {
        final userId = state.pathParameters['userId']!;
        return ProfileScreen(userId: userId);
      },
    ),
    ShellRoute(
      builder: (context, state, child) => ScaffoldWithNav(child: child),
      routes: [
        GoRoute(path: '/feed', builder: (_, __) => const FeedScreen()),
        GoRoute(path: '/settings', builder: (_, __) => const SettingsScreen()),
      ],
    ),
  ],
);''',
]


# ------------------------------------------------------------------
# GitHub code search (public API, no auth needed for basic search)
# ------------------------------------------------------------------

def fetch_github_code_snippets(
    query: str,
    language: str,
    max_snippets: int = 100,
) -> list[str]:
    """Fetch code from GitHub search API (limited without auth)."""
    snippets: list[str] = []
    page = 1

    while len(snippets) < max_snippets:
        params = urllib.parse.urlencode({
            "q": f"{query} language:{language}",
            "per_page": "30",
            "page": str(page),
        })
        url = f"https://api.github.com/search/code?{params}"
        req = urllib.request.Request(url, headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/vnd.github.v3.text-match+json",
        })

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                # Extract text matches (code fragments).
                for match in item.get("text_matches", []):
                    fragment = match.get("fragment", "").strip()
                    if len(fragment) > 50:
                        # Add file context.
                        name = item.get("name", "unknown")
                        snippet = f"# File: {name}\n{fragment}"
                        snippets.append(snippet)

            log.info("GitHub: fetched %d snippets (page %d)", len(snippets), page)
            page += 1

            if page > 5:  # GitHub rate-limits heavily without auth.
                break

        except Exception as e:
            log.warning("GitHub fetch failed: %s", e)
            break

    return snippets[:max_snippets]


# ------------------------------------------------------------------
# Save corpus
# ------------------------------------------------------------------

def save_code_corpus(snippets: list[str], path: Path) -> None:
    """Save code snippets as one-per-line corpus."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for snippet in snippets:
            # Replace newlines within snippet with special separator.
            line = snippet.replace("\n", " \\n ").strip()
            if line:
                f.write(line + "\n")

    total_chars = sum(len(s) for s in snippets)
    log.info("Saved %d snippets (%d chars) to %s", len(snippets), total_chars, path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare code corpus")
    parser.add_argument("--lang", default="all", choices=["python", "dart", "all"])
    parser.add_argument("--max-snippets", type=int, default=500,
                        help="Max GitHub snippets per language")
    parser.add_argument("--no-github", action="store_true",
                        help="Skip GitHub fetch, use only built-in patterns")
    args = parser.parse_args()

    all_snippets: list[str] = []

    if args.lang in ("python", "all"):
        log.info("Adding %d built-in Python patterns", len(PYTHON_PATTERNS))
        all_snippets.extend(PYTHON_PATTERNS)

        if not args.no_github:
            for query in ["def ", "class ", "async def ", "import numpy"]:
                gh = fetch_github_code_snippets(query, "python", args.max_snippets // 4)
                all_snippets.extend(gh)

    if args.lang in ("dart", "all"):
        log.info("Adding %d built-in Dart patterns", len(DART_PATTERNS))
        all_snippets.extend(DART_PATTERNS)

        if not args.no_github:
            for query in ["Widget build", "StatefulWidget", "Future<"]:
                gh = fetch_github_code_snippets(query, "dart", args.max_snippets // 3)
                all_snippets.extend(gh)

    if not all_snippets:
        log.error("No code snippets collected!")
        return

    # Duplicate patterns 5x to reinforce learning.
    builtin_count = len(PYTHON_PATTERNS) + len(DART_PATTERNS)
    for _ in range(4):
        all_snippets.extend(PYTHON_PATTERNS)
        all_snippets.extend(DART_PATTERNS)

    output = DATA_DIR / "code_corpus.txt"
    save_code_corpus(all_snippets, output)

    log.info(
        "Done! %d total snippets (%d built-in × 5, rest from GitHub)",
        len(all_snippets), builtin_count,
    )
    log.info("Corpus ready at: %s", output)


if __name__ == "__main__":
    main()
