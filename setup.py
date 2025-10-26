#!/usr/bin/env python3
"""
Setup script for PRISM: Privacy-Preserving Human Action Recognition
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="prism-action-recognition",
    version="1.0.0",
    author="PRISM Development Team",
    author_email="prism@example.com",
    description="Privacy-Preserving Human Action Recognition via ε-Differential Private Spatial-Temporal Graph Networks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/prism",
    project_urls={
        "Bug Reports": "https://github.com/your-username/prism/issues",
        "Source": "https://github.com/your-username/prism",
        "Documentation": "https://github.com/your-username/prism/docs",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "prism-train=examples.train_example:main",
            "prism-benchmark=examples.benchmark_example:main",
            "prism-optimize=examples.optimization_example:main",
        ],
    },
    include_package_data=True,
    package_data={
        "prism": ["data/*.json", "models/*.yaml", "configs/*.yaml"],
    },
    keywords=[
        "privacy-preserving",
        "differential-privacy",
        "action-recognition",
        "graph-neural-networks",
        "pose-analysis",
        "healthcare-ai",
        "spatial-temporal",
        "machine-learning",
    ],
    zip_safe=False,
)
