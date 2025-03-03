from setuptools import setup, find_packages

setup(
    name="dnallm",
    version="0.1.0",
    description="A toolkit for fine-tuning and inference with DNA Language Models",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "transformers>=4.15.0",
        "datasets>=1.18.0",
        "scikit-learn>=0.24.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "click>=7.0",
        "wandb>=0.12.0",
        "jax>=0.3.0",  # For Nucleotide Transformer
        "haiku>=0.0.5", # For Nucleotide Transformer
        "modelscope>=1.9.0",  # Add ModelScope support
    ],
    entry_points={
        "console_scripts": [
            "dnallm-train=dnallm.cli.train:main",
            "dnallm-predict=dnallm.cli.predict:main",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    extras_require={
        'test': [
            'pytest>=6.0.0',
            'pytest-cov>=2.0.0',
        ],
    },
) 