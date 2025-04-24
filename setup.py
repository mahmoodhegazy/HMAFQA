# setup.py
from setuptools import setup, find_packages

setup(
    name="hmafqa",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "sentence-transformers",
        "openai",
        "sympy",
        "pandas",
        "numpy",
        "scikit-learn",
    ],
    entry_points={
        "console_scripts": [
            "hmafqa=hmafqa.scripts.cli:main",
        ],
    },
    python_requires=">=3.7",
    author="Hegazy",
    description="Hybrid Multi-Agent Framework for Financial QA",
    license="MIT",
)