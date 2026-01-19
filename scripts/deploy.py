import argparse
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from datetime import datetime


def deploy_model(
    model_package_arn=None,
    model_data=None,
    endpoint_name=None,
    instance_type="ml.m5.xlarge",
    initial_instance_count=1,
    role=None,
    region="us-east-1"
):
    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    
    if not role:
        role = sagemaker.get_execution_role()
    
    if not endpoint_name:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        endpoint_name = f"wine-quality-endpoint-{timestamp}"
    
    print(f"Deploying model to endpoint: {endpoint_name}")
    
    if model_package_arn:
        print(f"Using model package: {model_package_arn}")
        
        sm_client = boto3.client('sagemaker', region_name=region)
        
        model_name = f"wine-quality-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        create_model_response = sm_client.create_model(
            ModelName=model_name,
            Containers=[{
                'ModelPackageName': model_package_arn
            }],
            ExecutionRoleArn=role
        )
        
        endpoint_config_name = f"{endpoint_name}-config"
        
        sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InstanceType': instance_type,
                'InitialInstanceCount': initial_instance_count,
            }],
            DataCaptureConfig={
                'EnableCapture': True,
                'InitialSamplingPercentage': 100,
                'DestinationS3Uri': f's3://{sagemaker_session.default_bucket()}/data-capture',
                'CaptureOptions': [
                    {'CaptureMode': 'Input'},
                    {'CaptureMode': 'Output'}
                ]
            }
        )
        
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        print(f"Endpoint creation in progress...")
        
    elif model_data:
        print(f"Using model artifacts: {model_data}")
        
        model = SKLearnModel(
            model_data=model_data,
            role=role,
            entry_point="inference.py",
            source_dir="src",
            framework_version="1.2-1",
            sagemaker_session=sagemaker_session,
        )
        
        predictor = model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            data_capture_config=sagemaker.model_monitor.DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=100,
                destination_s3_uri=f's3://{sagemaker_session.default_bucket()}/data-capture'
            )
        )
        
        print(f"Model deployed successfully!")
        
    else:
        raise ValueError("Either model_package_arn or model_data must be provided")
    
    print(f"\nEndpoint Name: {endpoint_name}")
    print(f"Instance Type: {instance_type}")
    print(f"Instance Count: {initial_instance_count}")
    print(f"\nTest the endpoint with:")
    print(f"  python scripts/test_endpoint.py --endpoint-name {endpoint_name}")
    
    return endpoint_name


def get_latest_approved_model_package(model_package_group_name, region="us-east-1"):
    sm_client = boto3.client('sagemaker', region_name=region)
    
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )
    
    if response['ModelPackageSummaryList']:
        return response['ModelPackageSummaryList'][0]['ModelPackageArn']
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-package-group', type=str, default='wine-quality-models')
    parser.add_argument('--model-package-arn', type=str)
    parser.add_argument('--model-data', type=str)
    parser.add_argument('--endpoint-name', type=str)
    parser.add_argument('--instance-type', type=str, default='ml.m5.xlarge')
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument('--role', type=str)
    parser.add_argument('--region', type=str, default='us-east-1')
    
    args = parser.parse_args()
    
    role = args.role or os.environ.get('SAGEMAKER_ROLE_ARN')
    
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
            print(f"No approved model found in {args.model_package_group}")
            return
    
    deploy_model(
        model_package_arn=args.model_package_arn,
        model_data=args.model_data,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        initial_instance_count=args.instance_count,
        role=role,
        region=args.region
    )


if __name__ == "__main__":
    import os
    main()
