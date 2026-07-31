"""
Setup script for the Mixed Reward package.

Install:
    pip install -e .

Integration with verl:
    verl reads custom_reward_function.path from its Hydra config.
    Point it to: mixed_reward.reward_manager.multi_dimensional
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mixed-reward-rlhf",
    version="1.0.0",
    author="Mixed Reward Team",
    description='"标量+维度" Hybrid Reward/Punishment Mechanism for RLHF',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "scripts"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "wandb>=0.16.0",
        "tensorboard>=2.14.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "verl": [
            "verl",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
