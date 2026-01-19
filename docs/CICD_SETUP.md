# CI/CD Setup Guide

This guide explains how to set up the GitHub Actions CI/CD pipeline for SageMaker MLOps.

## Overview

The CI/CD pipeline includes three workflows:

1. **SageMaker Pipeline** (`sagemaker-pipeline.yml`) - Main CI/CD workflow
2. **Model Deployment** (`model-deploy.yml`) - Manual model deployment
3. **Cleanup** (`cleanup.yml`) - Delete SageMaker resources

## Workflow Details

### 1. SageMaker Pipeline Workflow

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main`
- Manual dispatch

**Jobs:**

#### Test Job
Runs on all triggers:
- Linting with flake8
- Unit tests with pytest
- Code quality checks

#### Build and Push Job
Runs on push/manual dispatch:
- Validates pipeline definition
- Checks AWS connectivity

#### Deploy Staging Job
Runs on push to `develop`:
- Creates/updates SageMaker pipeline
- Executes pipeline in staging environment

#### Deploy Production Job
Runs on push to `main`:
- Creates/updates SageMaker pipeline
- Deploys approved model to production endpoint
- Runs endpoint tests
- Uploads deployment artifacts

### 2. Model Deployment Workflow

**Trigger:** Manual workflow dispatch

**Inputs:**
- `model_package_arn`: ARN of model to deploy
- `endpoint_name`: Name for the endpoint
- `instance_type`: EC2 instance type (default: ml.m5.xlarge)
- `instance_count`: Number of instances (default: 1)
- `environment`: staging or production

**Steps:**
- Deploy model to specified endpoint
- Wait for endpoint to be ready
- Run endpoint tests
- Generate deployment summary

### 3. Cleanup Workflow

**Trigger:** Manual workflow dispatch

**Inputs:**
- `endpoint_name`: Endpoint to delete
- `confirm`: Type "DELETE" to confirm

**Steps:**
- Delete SageMaker endpoint
- Delete endpoint configuration

## GitHub Secrets Setup

### Required Secrets

Configure these in your GitHub repository settings (`Settings` → `Secrets and variables` → `Actions`):

1. **AWS_ACCESS_KEY_ID**
   - AWS access key for GitHub Actions
   - Best practice: Create dedicated IAM user for CI/CD

2. **AWS_SECRET_ACCESS_KEY**
   - AWS secret access key

3. **SAGEMAKER_ROLE_ARN**
   - SageMaker execution role ARN
   - Example: `arn:aws:iam::123456789012:role/SageMakerExecutionRole`

4. **S3_BUCKET**
   - S3 bucket for SageMaker artifacts
   - Example: `my-sagemaker-bucket`

### Creating IAM User for GitHub Actions

```bash
# Create IAM user
aws iam create-user --user-name github-actions-sagemaker

# Create access key
aws iam create-access-key --user-name github-actions-sagemaker

# Attach policy (use inline policy below)
aws iam put-user-policy \
  --user-name github-actions-sagemaker \
  --policy-name SageMakerCICD \
  --policy-document file://github-actions-policy.json
```

**Policy Document** (`github-actions-policy.json`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreatePipeline",
        "sagemaker:UpdatePipeline",
        "sagemaker:StartPipelineExecution",
        "sagemaker:DescribePipeline",
        "sagemaker:DescribePipelineExecution",
        "sagemaker:ListPipelineExecutions",
        "sagemaker:CreateModel",
        "sagemaker:CreateEndpointConfig",
        "sagemaker:CreateEndpoint",
        "sagemaker:UpdateEndpoint",
        "sagemaker:DeleteEndpoint",
        "sagemaker:DeleteEndpointConfig",
        "sagemaker:DescribeEndpoint",
        "sagemaker:InvokeEndpoint",
        "sagemaker:ListModelPackages",
        "sagemaker:DescribeModelPackage",
        "sagemaker:CreateProcessingJob",
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeProcessingJob",
        "sagemaker:DescribeTrainingJob"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::YOUR_BUCKET_NAME",
        "arn:aws:s3:::YOUR_BUCKET_NAME/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "iam:PassRole"
      ],
      "Resource": "arn:aws:iam::*:role/SageMakerExecutionRole*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

### Creating SageMaker Execution Role

```bash
# Create trust policy
cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name SageMakerExecutionRole \
  --assume-role-policy-document file://trust-policy.json

# Attach managed policy
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Attach S3 access policy
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

## GitHub Environments Setup

### Create Environments

1. Go to `Settings` → `Environments`
2. Create two environments:
   - `staging`
   - `production`

### Configure Environment Protection Rules

**For Production:**
1. Enable "Required reviewers" (at least 1 reviewer)
2. Set deployment branch to `main` only
3. Add environment secrets if different from staging

**For Staging:**
1. Set deployment branch to `develop` only

## Usage

### Automatic Deployment

**Staging:**
```bash
git checkout develop
git add .
git commit -m "Update model"
git push origin develop
```
→ Triggers pipeline execution in staging

**Production:**
```bash
git checkout main
git merge develop
git push origin main
```
→ Triggers pipeline execution and endpoint deployment in production

### Manual Model Deployment

1. Go to `Actions` → `Model Deployment`
2. Click `Run workflow`
3. Fill in parameters:
   - Model Package ARN (from Model Registry)
   - Endpoint name
   - Instance type and count
   - Environment (staging/production)
4. Click `Run workflow`

### Cleanup Resources

1. Go to `Actions` → `Cleanup Resources`
2. Click `Run workflow`
3. Enter endpoint name
4. Type "DELETE" to confirm
5. Click `Run workflow`

## Monitoring

### View Pipeline Executions

GitHub Actions:
- Go to `Actions` tab in GitHub
- Select workflow run
- View job logs and artifacts

AWS Console:
- [SageMaker Pipelines](https://console.aws.amazon.com/sagemaker/home#/pipelines)
- [SageMaker Endpoints](https://console.aws.amazon.com/sagemaker/home#/endpoints)

### Artifacts

Each workflow run stores artifacts:
- `pipeline-execution-staging` - Staging pipeline info (30 days)
- `pipeline-execution-production` - Production pipeline info (90 days)

## Best Practices

1. **Branch Strategy**
   - `develop` → Staging environment
   - `main` → Production environment
   - Use pull requests for code review

2. **Model Approval**
   - Manually approve models in Model Registry before production deployment
   - Set accuracy thresholds in pipeline parameters

3. **Testing**
   - Always test in staging before production
   - Run unit tests locally before pushing

4. **Security**
   - Use least-privilege IAM policies
   - Rotate AWS access keys regularly
   - Never commit secrets to repository

5. **Cost Optimization**
   - Delete unused endpoints
   - Use spot instances for training (configure in pipeline)
   - Monitor S3 storage costs

## Troubleshooting

### Workflow fails with "Access Denied"
- Check IAM permissions
- Verify SageMaker execution role ARN
- Ensure S3 bucket exists and is accessible

### Pipeline execution fails
- Check CloudWatch logs for training/processing jobs
- Verify data exists in S3
- Check pipeline definition in SageMaker console

### Endpoint deployment fails
- Verify model is approved in Model Registry
- Check instance type availability in region
- Review CloudWatch logs for errors

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [SageMaker Pipelines Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [SageMaker MLOps Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/best-practices.html)
