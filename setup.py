from setuptools import setup, find_packages

setup(
    name="sagemaker-mlops-demo",
    version="0.1.0",
    description="SageMaker MLOps Demo with scikit-learn",
    author="Your Name",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "sagemaker>=2.200.0",
        "boto3>=1.28.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
)
