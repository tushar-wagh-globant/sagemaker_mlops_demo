import argparse
import boto3
import os


def upload_to_s3(local_file, s3_bucket, s3_key=None, region="us-east-1"):
    if not s3_key:
        s3_key = os.path.basename(local_file)
    
    s3_client = boto3.client('s3', region_name=region)
    
    print(f"Uploading {local_file} to s3://{s3_bucket}/{s3_key}")
    
    s3_client.upload_file(local_file, s3_bucket, s3_key)
    
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    print(f"Upload complete: {s3_uri}")
    
    return s3_uri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data/wine-quality.csv')
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--key', type=str, default='data/wine-quality.csv')
    parser.add_argument('--region', type=str, default='us-east-1')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        print("Run 'python scripts/download_data.py' first to download the dataset")
        return
    
    upload_to_s3(args.file, args.bucket, args.key, args.region)


if __name__ == "__main__":
    main()
