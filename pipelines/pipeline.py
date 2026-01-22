import os
import json
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat
from sagemaker.workflow.properties import PropertyFile


def get_pipeline(
    region,
    role,
    s3_bucket,
    pipeline_name="WineQualityPipeline",
    model_package_group_name="wine-quality-models",
    base_job_prefix="wine-quality",
):
    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    
    # Parameters
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
    input_data_uri = ParameterString(name="InputDataUri", default_value=f"s3://{s3_bucket}/data/wine-quality.csv")
    accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.70)
    
    # Processing Step
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=sagemaker_session,
    )
    
    step_process = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/output/test"),
        ],
        code="src/preprocess.py",
    )
    
    # Training Step
    sklearn_estimator = Estimator(
        image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
        instance_type=training_instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=sagemaker_session,
    )
    
    step_train = TrainingStep(
        name="TrainModel",
        estimator=sklearn_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )
    
    # Evaluation Step
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    
    eval_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        role=role,
        sagemaker_session=sagemaker_session,
    )
    
    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code="src/evaluate.py",
        property_files=[evaluation_report],
    )
    
    # Condition Step
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path="metrics.accuracy",
        ),
        right=accuracy_threshold,
    )
    
    step_cond = ConditionStep(
        name="CheckAccuracyThreshold",
        conditions=[cond_gte],
        if_steps=[],
        else_steps=[],
    )
    
    # Pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            training_instance_type,
            input_data_uri,
            accuracy_threshold,
        ],
        steps=[step_process, step_train, step_evaluate, step_cond],
        sagemaker_session=sagemaker_session,
    )
    
    return pipeline


def main():
    import yaml
    from pathlib import Path
    
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print("Loaded environment from .env file")
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    role = os.environ.get('SAGEMAKER_ROLE_ARN') or config['sagemaker']['role_arn']
    s3_bucket = os.environ.get('S3_BUCKET') or config['sagemaker']['s3_bucket']
    region = os.environ.get('AWS_REGION') or config['sagemaker']['region']
    
    if not role or '${' in role:
        raise ValueError("SAGEMAKER_ROLE_ARN environment variable not set")
    
    if not s3_bucket or '${' in s3_bucket:
        raise ValueError("S3_BUCKET environment variable not set")
    
    print(f"Creating pipeline...")
    print(f"Region: {region}")
    print(f"Role: {role}")
    print(f"S3 Bucket: {s3_bucket}")
    
    pipeline = get_pipeline(
        region=region,
        role=role,
        s3_bucket=s3_bucket,
    )
    
    print("\nUpserting pipeline...")
    pipeline.upsert(role_arn=role)
    
    print("\nStarting pipeline execution...")
    execution = pipeline.start()
    
    print(f"Pipeline execution started: {execution.arn}")
    print(f"\nView pipeline in AWS Console:")
    print(f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/pipelines")
    
    return execution


if __name__ == "__main__":
    main()
