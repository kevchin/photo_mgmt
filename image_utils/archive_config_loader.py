#!/usr/bin/env python3
"""
Configuration loader for photo archives.

Loads archive configurations from YAML file and provides
utilities for managing multiple photo databases.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass


@dataclass
class ArchiveConfig:
    """Configuration for a single photo archive"""
    name: str
    id: str
    db_connection: str
    root_dir: str
    description: str = ""
    llm_model: str = ""  # Optional per-archive LLM model override
    caption_detail: str = "basic"  # Caption detail level
    
    def get_root_path(self) -> Path:
        """Get expanded root directory path"""
        return Path(os.path.expanduser(self.root_dir))


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "all-MiniLM-L6-v2"
    dimensions: int = 384


@dataclass
class LLMConfig:
    """Local LLM configuration for caption generation"""
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    max_tokens: int = 500


@dataclass
class GlobalConfig:
    """Global configuration containing all archives and settings"""
    archives: List[ArchiveConfig]
    default_archive_id: str
    embedding: EmbeddingConfig
    llm: LLMConfig
    
    def get_archive(self, archive_id: str) -> Optional[ArchiveConfig]:
        """Get archive configuration by ID"""
        for archive in self.archives:
            if archive.id == archive_id:
                return archive
        return None
    
    def get_default_archive(self) -> Optional[ArchiveConfig]:
        """Get default archive configuration"""
        return self.get_archive(self.default_archive_id)
    
    def get_archive_names(self) -> Dict[str, str]:
        """Get mapping of archive IDs to names for UI selection"""
        return {archive.id: archive.name for archive in self.archives}


DEFAULT_CONFIG_PATH = Path(__file__).parent / "archives_config.yaml"


def load_config(config_path: Optional[Path] = None) -> GlobalConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
        
    Returns:
        GlobalConfig object with all settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        ValueError: If required fields are missing
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Parse archives
    archives = []
    for arch_data in data.get('archives', []):
        archive = ArchiveConfig(
            name=arch_data.get('name', ''),
            id=arch_data.get('id', ''),
            db_connection=arch_data.get('db_path', ''),  # Support both db_path and db_connection
            root_dir=arch_data.get('root_dir', ''),
            description=arch_data.get('description', ''),
            llm_model=arch_data.get('llm_model', ''),
            caption_detail=arch_data.get('caption_detail', 'basic')
        )
        archives.append(archive)
    
    if not archives:
        raise ValueError("No archives defined in configuration")
    
    # Parse embedding config
    emb_data = data.get('embedding', {})
    embedding = EmbeddingConfig(
        model_name=emb_data.get('model_name', 'all-MiniLM-L6-v2'),
        dimensions=emb_data.get('dimensions', 384)
    )
    
    # Parse LLM config
    llm_data = data.get('llm', {})
    llm = LLMConfig(
        provider=llm_data.get('provider', 'ollama'),
        base_url=llm_data.get('base_url', 'http://localhost:11434'),
        model=llm_data.get('model', 'llama3.2'),
        max_tokens=llm_data.get('max_tokens', 500)
    )
    
    # Get default archive ID
    default_id = data.get('default_archive_id', archives[0].id)
    
    # Check for settings section with defaults
    settings_data = data.get('settings', {})
    if settings_data.get('default_llm_model') and not llm.model:
        llm.model = settings_data['default_llm_model']
    if settings_data.get('default_caption_detail'):
        # This is a global default, can be overridden per archive
        pass
    
    return GlobalConfig(
        archives=archives,
        default_archive_id=default_id,
        embedding=embedding,
        llm=llm
    )


def find_config_file(search_paths: Optional[List[Path]] = None) -> Optional[Path]:
    """
    Find config file in common locations.
    
    Args:
        search_paths: Additional paths to search
        
    Returns:
        Path to config file if found, None otherwise
    """
    paths_to_check = [
        DEFAULT_CONFIG_PATH,
        Path.home() / ".config" / "photo_archive" / "archives_config.yaml",
        Path("/etc/photo_archive/archives_config.yaml"),
    ]
    
    if search_paths:
        paths_to_check.extend(search_paths)
    
    for path in paths_to_check:
        if path.exists():
            return path
    
    return None


def save_config(config: GlobalConfig, config_path: Path):
    """
    Save configuration to YAML file.
    
    Args:
        config: GlobalConfig object to save
        config_path: Path to save to
    """
    data = {
        'archives': [
            {
                'name': arch.name,
                'id': arch.id,
                'db_connection': arch.db_connection,
                'root_dir': arch.root_dir,
                'description': arch.description
            }
            for arch in config.archives
        ],
        'default_archive_id': config.default_archive_id,
        'embedding': {
            'model_name': config.embedding.model_name,
            'dimensions': config.embedding.dimensions
        },
        'llm': {
            'provider': config.llm.provider,
            'base_url': config.llm.base_url,
            'model': config.llm.model,
            'max_tokens': config.llm.max_tokens
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def create_sample_config(output_path: Path):
    """Create a sample configuration file"""
    sample = GlobalConfig(
        archives=[
            ArchiveConfig(
                name="Production Archive",
                id="prod_v1",
                db_connection="postgresql://user:password@localhost:5432/photo_archive",
                root_dir="~/Documents/photos1",
                description="Main production photo archive"
            ),
            ArchiveConfig(
                name="Test Archive",
                id="test_v1",
                db_connection="postgresql://user:password@localhost:5432/photo_test",
                root_dir="~/Downloads/test_photos",
                description="Testing new captioning models"
            )
        ],
        default_archive_id="prod_v1",
        embedding=EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            dimensions=384
        ),
        llm=LLMConfig(
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3.2",
            max_tokens=500
        )
    )
    
    save_config(sample, output_path)
    print(f"Sample config created at: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Photo Archive Configuration Utility")
    parser.add_argument('--show', action='store_true', help='Show current configuration')
    parser.add_argument('--create-sample', type=Path, help='Create sample config at specified path')
    parser.add_argument('--config', type=Path, help='Path to config file')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_config(args.create_sample)
    else:
        try:
            config = load_config(args.config)
            print(f"Loaded {len(config.archives)} archives:")
            for arch in config.archives:
                marker = " (default)" if arch.id == config.default_archive_id else ""
                print(f"  - {arch.name}{marker}")
                print(f"    ID: {arch.id}")
                print(f"    DB: {arch.db_connection}")
                print(f"    Root: {arch.root_dir}")
                if arch.description:
                    print(f"    Description: {arch.description}")
                print()
            
            print(f"Embedding model: {config.embedding.model_name} ({config.embedding.dimensions} dims)")
            print(f"LLM: {config.llm.provider}/{config.llm.model}")
        except Exception as e:
            print(f"Error loading config: {e}")
