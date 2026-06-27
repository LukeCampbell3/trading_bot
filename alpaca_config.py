"""
Alpaca Trading API Configuration
Free Paper Trading API Setup
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file.
# `override=True` ensures the local .env wins over stale IDE/shell env vars.
load_dotenv(override=True)

class AlpacaConfig:
    """Configuration for Alpaca Paper Trading (Free API)"""
    
    # Get credentials from environment variables (recommended)
    API_KEY = os.getenv('ALPACA_API_KEY', '')
    API_SECRET = os.getenv('ALPACA_API_SECRET', '')
    
    # Explicit endpoints (override in .env if needed)
    BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
    DATA_URL = os.getenv('ALPACA_DATA_URL', 'https://data.alpaca.markets').rstrip('/')
    
    # Explicitly select paper vs live trading endpoint behavior
    PAPER = os.getenv('ALPACA_PAPER', 'true').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    
    @classmethod
    def validate(cls):
        """Validate that credentials are set"""
        if not cls.API_KEY or not cls.API_SECRET:
            raise ValueError(
                "Alpaca API credentials not found!\n"
                "Please set ALPACA_API_KEY and ALPACA_API_SECRET in .env file\n"
                "Get free paper trading keys at: https://alpaca.markets/docs/trading/paper-trading/"
            )
        if not cls.BASE_URL.startswith("http"):
            raise ValueError("ALPACA_BASE_URL must be a valid URL (include http/https)")
        if not cls.DATA_URL.startswith("http"):
            raise ValueError("ALPACA_DATA_URL must be a valid URL (include http/https)")
        return True

# Validate on import
if __name__ != "__main__":
    try:
        AlpacaConfig.validate()
        print("✓ Alpaca configuration loaded successfully")
    except ValueError as e:
        print(f"⚠ Warning: {e}")
