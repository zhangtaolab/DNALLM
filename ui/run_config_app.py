#!/usr/bin/env python3
"""
Simple launcher for DNALLM Configuration Generator Gradio App

This script provides a simple way to launch the configuration generator web
interface without complex import path issues.
"""

import sys
import argparse
import os

os.environ["GRADIO_TEMP_DIR"] = "tmp/gradio"


def main():
    """Main function to launch the app"""
    parser = argparse.ArgumentParser(
        description="Launch DNALLM Configuration Generator Gradio App"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1, localhost only)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the server to (default: 7860)",
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link for the app (default: False)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (default: False)",
    )

    args = parser.parse_args()

    print("🚀 Launching DNALLM Configuration Generator...")
    print(
        f"📱 Web interface will be available at: http://{args.host}:{args.port}"
    )

    if args.share:
        print("🌐 Public link will be generated for sharing")

    try:
        # Import and launch the app
        from model_config_generator_app import GradioConfigGenerator

        generator = GradioConfigGenerator()
        interface = generator.create_interface()

        # Launch with custom settings
        launch_kwargs = {
            "server_name": args.host,
            "server_port": args.port,
            "share": args.share,
            "show_error": True,
        }

        # Add debug parameter only if supported
        if args.debug:
            launch_kwargs["debug"] = True

        interface.launch(**launch_kwargs)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure you have installed the required dependencies:")
        print("pip install gradio pyyaml")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
