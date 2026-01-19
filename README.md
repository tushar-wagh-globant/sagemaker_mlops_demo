# SageMaker MLOps Demo - Wine Quality Classification

A complete MLOps project demonstrating end-to-end machine learning workflow on AWS SageMaker with scikit-learn.

## Overview

This project demonstrates:
- ✅ Data preprocessing and feature engineering
- ✅ Model training with scikit-learn (Random Forest Classifier)
- ✅ Model evaluation and metrics tracking
- ✅ SageMaker Pipelines for workflow orchestration
- ✅ Model Registry integration
- ✅ SageMaker endpoint deployment
- ✅ Model monitoring and logging
- ✅ CI/CD integration (optional)

## Project Structure

```
sagemaker_mlops_demo/
├── src/
│   ├── train.py              # Training script
│   ├── inference.py          # Inference handler for SageMaker endpoint
│   ├── preprocess.py         # Data preprocessing
│   └── evaluate.py           # Model evaluation
├── pipelines/
│   └── pipeline.py           # SageMaker Pipeline definition
├── notebooks/
│   └── explore.ipynb         # Exploratory data analysis
├── data/
│   └── wine-quality.csv      # Sample dataset
├── config/
│   └── config.yaml           # Configuration file
├── tests/
│   └── test_model.py         # Unit tests
├── scripts/
│   ├── deploy.py             # Deployment script
│   └── setup_iam.py          # IAM role setup helper
├── pyproject.toml
└── README.md
```

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** configured
3. **Python 3.9+**
4. **SageMaker execution role** with necessary permissions

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

```bash
aws configure
```

### 3. Set Up Environment Variables

```bash
export SAGEMAKER_ROLE_ARN="arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole"
export S3_BUCKET="your-sagemaker-bucket"
export AWS_REGION="us-east-1"
```

### 4. Run the Pipeline

```bash
# Option A: Run complete SageMaker Pipeline
python pipelines/pipeline.py

# Option B: Run individual components
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

### 5. Deploy Model to Endpoint

```bash
python scripts/deploy.py --model-name wine-quality-model
```

## MLOps Workflow

### 1. Data Preparation
- Load wine quality dataset
- Feature engineering and scaling
- Train/test split
- Upload to S3

### 2. Training
- Train Random Forest Classifier
- Hyperparameter tuning (optional)
- Save model artifacts to S3

### 3. Evaluation
- Calculate metrics (accuracy, precision, recall, F1)
- Generate classification report
- Save metrics to model registry

### 4. Model Registration
- Register model in SageMaker Model Registry
- Version tracking
- Metadata and metrics attachment

### 5. Deployment
- Create SageMaker endpoint configuration
- Deploy model to real-time endpoint
- Enable data capture for monitoring

### 6. Monitoring
- CloudWatch metrics
- Model quality monitoring
- Data drift detection

## SageMaker Pipeline Steps

The pipeline includes:

1. **ProcessingStep**: Data preprocessing and feature engineering
2. **TrainingStep**: Model training with SageMaker training job
3. **EvaluationStep**: Model evaluation and metrics calculation
4. **ConditionStep**: Conditional model registration based on metrics threshold
5. **RegisterModelStep**: Register approved model to Model Registry

## Usage Examples

### Training Locally (for testing)

```bash
python src/train.py --local
```

### Invoke Endpoint

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

payload = {
    "instances": [
        [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
    ]
}

response = runtime.invoke_endpoint(
    EndpointName='wine-quality-endpoint',
    ContentType='application/json',
    Body=json.dumps(payload)
)

result = json.loads(response['Body'].read().decode())
print(f"Prediction: {result}")
```

### Clean Up Resources

```bash
python scripts/cleanup.py
```

## Configuration

Edit `config/config.yaml` to customize:
- Model hyperparameters
- Instance types
- S3 paths
- Pipeline settings

## Testing

```bash
pytest tests/
```

## CI/CD Integration

GitHub Actions workflow (`.github/workflows/sagemaker-pipeline.yml`) for:
- Automated testing
- Pipeline execution on merge
- Model deployment to staging/production

## Monitoring and Logging

- **CloudWatch Logs**: Training and endpoint logs
- **Model Monitor**: Data quality and model quality monitoring
- **Metrics**: Custom CloudWatch metrics for model performance

## Cost Optimization Tips

1. Use spot instances for training
2. Delete endpoints when not in use
3. Use SageMaker Serverless Inference for low-traffic endpoints
4. Enable auto-scaling for endpoints

## Troubleshooting

### Common Issues

**Issue**: "Unable to locate credentials"
```bash
aws configure
```

**Issue**: "Access Denied to S3"
- Check SageMaker execution role has S3 permissions

**Issue**: "Endpoint creation failed"
- Check CloudWatch logs for detailed error messages

## Next Steps

- [ ] Add hyperparameter tuning with SageMaker Tuner
- [ ] Implement A/B testing with multiple model variants
- [ ] Add feature store integration
- [ ] Set up model monitoring dashboards
- [ ] Implement automated retraining pipeline

## Resources

- [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)

## License

MIT
