"""
Application configuration for the Sentiment Model Arena.
"""

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or defaults.
    """
    
    # Hugging Face Hub repository IDs
    SCRATCH_REPO_ID: str = "jibinsajujoseph/scratch-transformer"
    DISTILROBERTA_REPO_ID: str = "jibinsajujoseph/distilroberta-imdb"

settings = Settings()
