from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mlxrd",
    version="2.0.0",
    author="ML XRD Team",
    description="Complete Machine Learning Pipeline for XRD Data Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml_xrd",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "tqdm>=4.60.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "all": [
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "optuna>=3.0.0",
            "plotly>=5.0.0",
            "shap>=0.41.0",
            "psutil>=5.8.0",
            "openpyxl>=3.0.0",
        ],
        "ml": [
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "optuna>=3.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "shap>=0.41.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlxrd=mlxrd.utils.cli:main",
        ],
    },
)
