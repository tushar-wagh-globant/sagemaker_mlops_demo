import argparse
import boto3


def delete_endpoint(endpoint_name, region="us-east-1"):
    sm_client = boto3.client('sagemaker', region_name=region)
    
    try:
        endpoint_desc = sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = endpoint_desc['EndpointConfigName']
        
        print(f"Deleting endpoint: {endpoint_name}")
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        
        print(f"Deleting endpoint config: {endpoint_config_name}")
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        
        print("Cleanup complete!")
        
    except sm_client.exceptions.ClientError as e:
        print(f"Error: {e}")


def list_endpoints(region="us-east-1"):
    sm_client = boto3.client('sagemaker', region_name=region)
    
    response = sm_client.list_endpoints(
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=50
    )
    
    print("\n=== SageMaker Endpoints ===")
    for endpoint in response['Endpoints']:
        print(f"Name: {endpoint['EndpointName']}")
        print(f"  Status: {endpoint['EndpointStatus']}")
        print(f"  Created: {endpoint['CreationTime']}")
        print()
    
    return response['Endpoints']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint-name', type=str, help='Endpoint name to delete')
    parser.add_argument('--list', action='store_true', help='List all endpoints')
    parser.add_argument('--region', type=str, default='us-east-1')
    
    args = parser.parse_args()
    
    if args.list:
        list_endpoints(args.region)
    elif args.endpoint_name:
        delete_endpoint(args.endpoint_name, args.region)
    else:
        print("Please specify --endpoint-name or --list")


if __name__ == "__main__":
    main()
