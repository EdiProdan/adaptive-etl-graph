#!/usr/bin/env python3
"""
Intelligent System for Semantic Structuring of Multimodal Data
Entry point for the application
"""
import argparse
import os
import yaml
from src.pipeline.controller import PipelineController


def parse_args():
    parser = argparse.ArgumentParser(
        description='Intelligent System for Semantic Structuring of Multimodal Data'
    )
    parser.add_argument(
        '-c', '--config',
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input file or directory path'
    )
    parser.add_argument(
        '-o', '--output',
        default='data/processed',
        help='Output directory for processed data'
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    config = load_config(args.config)

    controller = PipelineController(config)

    if os.path.isdir(args.input):
        for filename in os.listdir(args.input):
            file_path = os.path.join(args.input, filename)
            if os.path.isfile(file_path):
                print(f"Processing {file_path}...")
                result = controller.process(file_path)
                # Save or use result as needed
                print(result)
    else:
        # Process single file
        print(f"Processing {args.input}...")
        result = controller.process(args.input)
        # Save or use result as needed
        print(result)

    print("Processing complete.")


if __name__ == "__main__":
    main()
