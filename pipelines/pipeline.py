import os
import json
import boto3
import sagemaker
import time
from sagemaker.sklearn.estimator import SKLearn
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
from sagemaker.model_metrics import MetricsSource, ModelMetrics 

def ensure_model_package_group_exists(group_name, region):
    sm_client = boto3.client("sagemaker", region_name=region)
    try:
        sm_client.describe_model_package_group(ModelPackageGroupName=group_name)
        print(f"✅ Group '{group_name}' already exists.")
    except sm_client.exceptions.ResourceNotFound:
        print(f"⚠️ Group '{group_name}' not found. Creating it...")
        sm_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription="Model group for Wine Quality pipeline"
        )
        time.sleep(2)
        print(f"✅ Group '{group_name}' created successfully.")

def get_pipeline(
    region,
    role,
    s3_bucket,
    pipeline_name="WineQualityPipeline",
    model_package_group_name="wine-quality-models",
    base_job_prefix="wine-quality",
):
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(
        boto_session=boto_session,
        default_bucket=s3_bucket
    )
    
    # --- Parameters ---
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.t3.medium")
    # ✅ FIX: Use ml.m5.large for training (T3 is not supported)
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.large")
    input_data_uri = ParameterString(name="InputDataUri", default_value=f"s3://{s3_bucket}/data/wine-quality.csv")
    accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.70)
    
    # --- Processing Step ---
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
    
    # --- Training Step ---
    # ✅ FIX: Use SKLearn Estimator instead of generic Estimator
    sklearn_estimator = SKLearn(
        entry_point="train.py",
        source_dir="src",
        framework_version="1.2-1",
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
    
    # --- Evaluation Step ---
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    
    eval_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.t3.medium",
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

    # --- Register Model Step ---
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"s3://{s3_bucket}/{base_job_prefix}/evaluation/evaluation.json",
            content_type="application/json"
        )
    )

    step_register = RegisterModel(
        name="RegisterWineQualityModel",
        estimator=sklearn_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv", "application/json"],
        response_types=["text/csv", "application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status="Approved",
        model_metrics=model_metrics,
    )
    
    # --- Condition Step ---
# --- Condition Step ---
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path="metrics.accuracy.value",  # <--- UPDATED HERE
        ),
        right=accuracy_threshold,
    )
    
    step_cond = ConditionStep(
        name="CheckAccuracyThreshold",
        conditions=[cond_gte],
        if_steps=[step_register], 
        else_steps=[],
    )
    
    # --- Pipeline ---
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
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    role = os.environ.get('SAGEMAKER_ROLE_ARN') or config['sagemaker']['role_arn']
    s3_bucket = os.environ.get('S3_BUCKET') or config['sagemaker']['s3_bucket']
    region = os.environ.get('AWS_REGION') or config['sagemaker']['region']
    
    # ✅ FIX: Ensure Group Exists before defining pipeline
    model_package_group_name = "wine-quality-models"
    ensure_model_package_group_exists(model_package_group_name, region)
    
    pipeline = get_pipeline(
        region=region,
        role=role,
        s3_bucket=s3_bucket,
        model_package_group_name=model_package_group_name
    )
    
    print("Upserting pipeline...")
    pipeline.upsert(role_arn=role)
    print("Starting execution...")
    execution = pipeline.start()
    print(f"Pipeline execution started: {execution.arn}")
    
    # ✅ FIX: Wait for execution so GitHub Action doesn't exit early
    try:
        execution.wait()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()