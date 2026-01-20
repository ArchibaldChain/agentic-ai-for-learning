"""
Configuration management for the agentic AI system.

This module handles loading configuration from environment variables,
with support for .env files using python-dotenv.

Example usage:
    from agentic_ai.utils.config import get_config

    config = get_config()
    print(config.aws_region)
    print(config.bedrock_model_id)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """
    Configuration container for AWS and model settings.

    Attributes:
        aws_region: AWS region for Bedrock API calls (default: us-east-1)
        aws_access_key_id: AWS access key (optional if using IAM roles)
        aws_secret_access_key: AWS secret key (optional if using IAM roles)
        bedrock_model_id: Claude model ID for text generation
        embedding_model_id: Titan model ID for embeddings
    """
    aws_region: str
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]
    bedrock_model_id: str
    embedding_model_id: str


def get_config(env_file: Optional[str] = None) -> Config:
    """
    Load configuration from environment variables.

    This function loads environment variables from a .env file (if present)
    and returns a Config object with all necessary settings.

    Args:
        env_file: Optional path to .env file. If not provided, looks for .env
                  in the current directory and parent directories.

    Returns:
        Config object with all settings populated.

    Raises:
        ValueError: If required environment variables are missing.

    Example:
        >>> config = get_config()
        >>> config.aws_region
        'us-east-1'
        >>> config.bedrock_model_id
        'anthropic.claude-3-5-sonnet-20241022-v2:0'
    """
    # Load .env file if it exists
    if env_file:
        load_dotenv(env_file)
    else:
        # Try to find .env file in current or parent directories
        current_dir = Path.cwd()
        for directory in [current_dir] + list(current_dir.parents):
            env_path = directory / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                break

    # Load configuration with defaults
    config = Config(
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        bedrock_model_id=os.getenv(
            "BEDROCK_MODEL_ID",
            "anthropic.claude-3-5-sonnet-20241022-v2:0"
        ),
        embedding_model_id=os.getenv(
            "EMBEDDING_MODEL_ID",
            "amazon.titan-embed-text-v1"
        ),
    )

    return config


# Singleton instance for convenience
_config: Optional[Config] = None


def get_default_config() -> Config:
    """
    Get or create the default configuration singleton.

    This is useful when you want to share configuration across modules
    without passing it explicitly.

    Returns:
        The shared Config instance.
    """
    global _config
    if _config is None:
        _config = get_config()
    return _config
