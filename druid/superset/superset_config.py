import os

# Database configuration
SQLALCHEMY_DATABASE_URI = 'postgresql://superset:superset@superset-db:5432/superset'

# Secret key for encryption - MUST be consistent
SECRET_KEY = os.environ.get('SUPERSET_SECRET_KEY', 'superduperset')

# Feature flags
FEATURE_FLAGS = {
    "ENABLE_TEMPLATE_PROCESSING": True,
}

# Cache configuration
CACHE_CONFIG = {
    'CACHE_TYPE': 'simple'
}

# Disable encryption for database connections if you don't need it
SQLALCHEMY_UTILS_DATABASE_ENCRYPTION_ENABLED = False

# Optional: Set a fixed encryption key
# SQLALCHEMY_UTILS_DATABASE_ENCRYPTION_KEY = 'your-32-character-encryption-key-here'

# Public role permissions
PUBLIC_ROLE_LIKE = "Gamma"

# Enable public registration
AUTH_USER_REGISTRATION = True
AUTH_USER_REGISTRATION_ROLE = "Public"