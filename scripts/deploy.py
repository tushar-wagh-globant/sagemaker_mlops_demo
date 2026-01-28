import argparse
import boto3
import sagemaker
import os
import sys
from sagemaker.sklearn.model import SKLearnModel
from datetime import datetime

def deploy_model(
    model_package_arn=None,
    model_data=None,
    endpoint_name=None,
    instance_type="ml.t2.medium",
    initial_instance_count=1,
    role=None,
    region="us-east-1",
    s3_bucket=None
):
    boto_session = boto3.Session(region_name=region)
    
    # Session setup with explicit bucket
    sagemaker_session = sagemaker.Session(
        boto_session=boto_session,
        default_bucket=s3_bucket
    )
    
    if not role:
        role = sagemaker.get_execution_role()
    
    # 1. Setup Naming
    # We use timestamps to ensure the Model and Config are unique.
    # This is REQUIRED to update an existing endpoint (Blue/Green deployment).
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    if not endpoint_name:
        endpoint_name = "wine-quality-endpoint"
    
    # Config name MUST be unique for every update
    endpoint_config_name = f"{endpoint_name}-config-{timestamp}"
    model_name = f"wine-quality-model-{timestamp}"

    print(f"🚀 Deployment Target: {endpoint_name}")
    print(f"📄 New Config Name:  {endpoint_config_name}")
    print(f"🪣 Bucket: {sagemaker_session.default_bucket()}")
    
    sm_client = boto3.client('sagemaker', region_name=region)
    
    if model_package_arn:
        print(f"📦 Using Model Package: {model_package_arn}")
        
        # 2. Create Model Entity (Immutable)
        print(f"   Creating Model Entity: {model_name}")
        sm_client.create_model(
            ModelName=model_name,
            Containers=[{
                'ModelPackageName': model_package_arn,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code'
                }
            }],
            ExecutionRoleArn=role
        )
        
        # 3. Create Endpoint Config (Immutable)
        # We DO NOT delete the old config; we just create a new unique one.
        print(f"   Creating Endpoint Config: {endpoint_config_name}")
        
        data_capture_config = {
            'EnableCapture': True,
            'InitialSamplingPercentage': 100,
            'DestinationS3Uri': f's3://{sagemaker_session.default_bucket()}/data-capture',
            'CaptureOptions': [{'CaptureMode': 'Input'}, {'CaptureMode': 'Output'}]
        }

        sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InstanceType': instance_type,
                'InitialInstanceCount': int(initial_instance_count),
            }],
            DataCaptureConfig=data_capture_config
        )
        
        # 4. Check & Update OR Create
        try:
            # Check if endpoint exists
            existing_endpoint = sm_client.describe_endpoint(EndpointName=endpoint_name)
            status = existing_endpoint['EndpointStatus']
            print(f"✅ Found existing endpoint '{endpoint_name}' (Status: {status})")
            
            

            if status not in ['InService', 'Failed']:
                print(f"⚠️ Endpoint is currently {status}. Cannot update immediately.")
                # Depending on needs, you might raise an error here
            
            print(f"🔄 Updating endpoint to use new config: {endpoint_config_name}")
            sm_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
        except sm_client.exceptions.ClientError:
            # If describe_endpoint fails, it doesn't exist.
            print(f"🆕 Endpoint '{endpoint_name}' does not exist. Creating new...")
            sm_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
        
        # 5. Wait for Deployment to Finish
        print(f"⏳ Waiting for endpoint '{endpoint_name}' to be InService...")
        waiter = sm_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        
    elif model_data:
        # Fallback for direct artifact deployment
        print(f"Using model artifacts: {model_data}")
        model = SKLearnModel(
            model_data=model_data,
            role=role,
            entry_point="inference.py",
            source_dir="src",
            framework_version="1.2-1",
            sagemaker_session=sagemaker_session,
        )
        model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )
        
    else:
        raise ValueError("Either model_package_arn or model_data must be provided")
    
    print(f"\n✅ Deployment Complete!")
    print(f"Endpoint Name: {endpoint_name}")
    return endpoint_name

def get_latest_approved_model_package(model_package_group_name, region="us-east-1"):
    sm_client = boto3.client('sagemaker', region_name=region)
    try:
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        if response['ModelPackageSummaryList']:
            return response['ModelPackageSummaryList'][0]['ModelPackageArn']
    except Exception as e:
        print(f"Error listing model packages: {e}")
        return None
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-package-group', type=str, default='wine-quality-models')
    parser.add_argument('--model-package-arn', type=str)
    parser.add_argument('--model-data', type=str)
    parser.add_argument('--endpoint-name', type=str)
    parser.add_argument('--instance-type', type=str, default='ml.t2.medium')
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument('--role', type=str)
    parser.add_argument('--region', type=str, default='us-east-1')
    
    args = parser.parse_args()
    
    role = args.role or os.environ.get('SAGEMAKER_ROLE_ARN')
    s3_bucket = os.environ.get('S3_BUCKET')
    
    if not s3_bucket:
        print("❌ Error: S3_BUCKET environment variable is missing.")
        sys.exit(1)
    
    if not args.model_package_arn and not args.model_data:
        print(f"No model specified, looking for latest approved model in {args.model_package_group}...")
        model_package_arn = get_latest_approved_model_package(
            args.model_package_group,
            region=args.region
        )
        
        if model_package_arn:
            print(f"Found model package: {model_package_arn}")
            args.model_package_arn = model_package_arn
        else:
            print(f"❌ Error: No approved model found in {args.model_package_group}")
            sys.exit(1)

    deploy_model(
        model_package_arn=args.model_package_arn,
        model_data=args.model_data,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        initial_instance_count=args.instance_count,
        role=role,
        region=args.region,
        s3_bucket=s3_bucket
    )

if __name__ == "__main__":
    main()