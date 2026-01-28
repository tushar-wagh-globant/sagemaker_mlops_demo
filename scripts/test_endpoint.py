import argparse
import json
import boto3
import sys
import numpy as np

def test_endpoint(endpoint_name, region="us-east-1"):
    # explicit session ensures we pick up the right credentials/region
    session = boto3.Session(region_name=region)
    runtime = session.client('sagemaker-runtime')
    
    # 1. Prepare Data
    # Matches the 11 features expected by the Wine Quality model
    sample_data = {
        "instances": [
            [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4], # Sample 1
            [7.8, 0.88, 0.0, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.2, 0.68, 9.8], # Sample 2
            [7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8] # Sample 3
        ]
    }
    
    print(f"🚀 Testing endpoint: {endpoint_name}")
    print(f"🌍 Region: {region}")
    print(f"📤 Sending payload with {len(sample_data['instances'])} records...")
    
    try:
        # 2. Invoke Endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Accept='application/json', # Explicitly ask for JSON back
            Body=json.dumps(sample_data)
        )
        
        # 3. Parse Response
        response_body = response['Body'].read().decode()
        result = json.loads(response_body)
        
        print("\n=== ✅ Prediction Results ===")
        print(f"Raw Response: {result}")
        
        # Friendly formatting
        if isinstance(result, list):
            for i, prediction in enumerate(result):
                print(f"  🍷 Wine {i+1}: Quality Prediction = {prediction}")
        
        return result

    except runtime.exceptions.ValidationError as e:
        print(f"\n❌ Validation Error: The endpoint rejected the data format.")
        print(f"Details: {e}")
        sys.exit(1)
        
    except runtime.exceptions.ModelError as e:
        print(f"\n❌ Model Error: The model failed to process the request.")
        print(f"Details: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error invoking endpoint: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint-name', type=str, required=True)
    parser.add_argument('--region', type=str, default='us-east-1')
    
    args = parser.parse_args()
    
    test_endpoint(args.endpoint_name, args.region)

if __name__ == "__main__":
    main()