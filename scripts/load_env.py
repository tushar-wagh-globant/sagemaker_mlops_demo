import os
from pathlib import Path


def load_env():
    env_file = Path(__file__).parent.parent / '.env'
    
    if not env_file.exists():
        print(f"Warning: .env file not found at {env_file}")
        print("Using environment variables or AWS CLI credentials")
        return
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if not os.environ.get(key):
                    os.environ[key] = value
    
    print("Environment variables loaded from .env file")


if __name__ == "__main__":
    load_env()
    
    print("\nCurrent Configuration:")
    print(f"AWS_REGION: {os.environ.get('AWS_REGION', 'Not set')}")
    print(f"SAGEMAKER_ROLE_ARN: {os.environ.get('SAGEMAKER_ROLE_ARN', 'Not set')}")
    print(f"S3_BUCKET: {os.environ.get('S3_BUCKET', 'Not set')}")
    print(f"AWS_ACCESS_KEY_ID: {'Set' if os.environ.get('AWS_ACCESS_KEY_ID') else 'Not set'}")
