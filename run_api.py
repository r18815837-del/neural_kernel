"""Entry point for running the Neural Kernel API server."""
from __future__ import annotations

import argparse
import logging

import uvicorn

from api.config import ApiConfig
from api.server import create_app


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration.

    Args:
        debug: Enable debug logging.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(
        description="Run Neural Kernel API server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on file changes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="build",
        help="Output directory for generated projects (default: build)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    args = parser.parse_args()

    setup_logging(debug=args.debug)

    # Create config
    config = ApiConfig(
        host=args.host,
        port=args.port,
        debug=args.debug,
        output_dir=args.output_dir,
    )

    # Create app
    app = create_app(config)

    # Run server
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=args.reload,
        workers=args.workers,
        log_level="debug" if args.debug else "info",
    )


if __name__ == "__main__":
    main()
