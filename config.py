"""
MindCare AI - Configuration File
=================================
Central configuration for the Flask application.
"""

import os

class Config:
    # Secret key for session management (change in production!)
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mindcare-ai-secret-key-2024-change-me')

    # Database path
    DATABASE = os.path.join(os.path.dirname(__file__), 'database', 'mental_health.db')

    # Model path
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'sentiment_model.pkl')

    # Debug mode (set False in production)
    DEBUG = os.environ.get('DEBUG', 'True') == 'True'

    # Maximum negative mood submissions before emergency warning
    NEGATIVE_MOOD_THRESHOLD = 3

    # App metadata
    APP_NAME = "MindCare AI"
    APP_VERSION = "1.0.0"
