import os
import sys
import pytest
from unittest import mock

# Add the parent directory to the path so we can import the config module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import Settings, get_settings


class TestSettings:
    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = Settings()
        assert settings.APP_NAME == "Mini-RAG"
        assert settings.DEBUG is False
        assert settings.VECTOR_DB_PATH == "vector_db"
    
    @mock.patch.dict(os.environ, {"DEBUG": "1"})
    def test_environment_override(self):
        """Test that environment variables override default settings."""
        settings = Settings()
        assert settings.DEBUG is True
    
    @mock.patch.dict(os.environ, {"VECTOR_DB_PATH": "/custom/path"})
    def test_vector_db_path_override(self):
        """Test that VECTOR_DB_PATH can be overridden."""
        settings = Settings()
        assert settings.VECTOR_DB_PATH == "/custom/path"


def test_get_settings():
    """Test the get_settings function returns a Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings) 