"""
NovaMind Setup Script
======================
Installation configuration for the NovaMind language model package.

Install in development mode:
    pip install -e .

Install normally:
    pip install .
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="novamind",
    version="0.1.0",
    author="Purushottam",
    author_email="",
    description="NovaMind: A custom Large Language Model built from scratch with pure PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/purushottam/novamind",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "tqdm",
        "matplotlib",
        "requests",
    ],
    extras_require={
        "dev": [
            "jupyter",
            "ipywidgets",
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "novamind-train=training.train:main",
            "novamind-collect=data.collector:collect_all",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
