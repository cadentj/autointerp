#!/usr/bin/env python3
"""
Launch script for the Autointerp Streamlit Dashboard

Usage:
    python run_dashboard.py /path/to/feature/directories [--port 8501] [--host localhost]
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Launch the Autointerp Streamlit Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_dashboard.py /workspace/cached_features
    python run_dashboard.py /workspace/cached_features --port 8502
    python run_dashboard.py /workspace/cached_features --host 0.0.0.0 --port 8501
        """
    )
    
    parser.add_argument(
        "features_path",
        help="Path to directory containing cached feature subdirectories"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run Streamlit on (default: 8501)"
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to run Streamlit on (default: localhost)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with additional logging"
    )
    
    args = parser.parse_args()
    
    # Validate features path
    if not os.path.exists(args.features_path):
        print(f"Error: Features path '{args.features_path}' does not exist")
        sys.exit(1)
    
    if not os.path.isdir(args.features_path):
        print(f"Error: Features path '{args.features_path}' is not a directory")
        sys.exit(1)
    
    # Get absolute path
    features_path = os.path.abspath(args.features_path)
    
    # Set up environment
    env = os.environ.copy()
    if args.debug:
        env["STREAMLIT_LOGGER_LEVEL"] = "debug"
    
    # Construct streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "app.py",
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--",
        features_path
    ]
    
    print(f"Starting Autointerp Dashboard...")
    print(f"Features path: {features_path}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()