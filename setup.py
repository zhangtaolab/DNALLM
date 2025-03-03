from setuptools import setup, find_packages

# Read requirements from files
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('requirements-dev.txt') as f:
    dev_requirements = f.read().splitlines()
    # Remove reference to requirements.txt
    dev_requirements.remove('-r requirements.txt')

setup(
    name="dnallm",
    version="0.1.0",
    description="A toolkit for fine-tuning and inference with DNA Language Models",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements,
        'test': [
            'pytest>=6.0.0',
            'pytest-cov>=2.0.0',
        ],
        'notebook': [
            'marimo>=0.1.0',
            'jupyter>=1.0.0',
        ]
    },
    entry_points={
        "console_scripts": [
            "dnallm-train=dnallm.cli.train:main",
            "dnallm-predict=dnallm.cli.predict:main",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
) 