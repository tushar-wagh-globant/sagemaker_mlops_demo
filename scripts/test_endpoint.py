import argparse
import json
import boto3
import numpy as np


def test_endpoint(endpoint_name, region="us-east-1"):
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    sample_data = {
        "instances": [
            [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
            [7.8, 0.88, 0.0, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.2, 0.68, 9.8],
            [7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8]
        ]
    }
    
    print(f"Testing endpoint: {endpoint_name}")
    print(f"Sample input: {json.dumps(sample_data, indent=2)}")
    
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(sample_data)
    )
    
    result = json.loads(response['Body'].read().decode())
    
    print("\n=== Prediction Results ===")
    print(json.dumps(result, indent=2))
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint-name', type=str, required=True)
    parser.add_argument('--region', type=str, default='us-east-1')
    
    args = parser.parse_args()
    
    test_endpoint(args.endpoint_name, args.region)


if __name__ == "__main__":
    main()
