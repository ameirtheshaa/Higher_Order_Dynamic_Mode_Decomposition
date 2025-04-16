from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tensor-dmd",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Tensor-based Dynamic Mode Decomposition for analyzing dynamical systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tensor-dmd",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "tensorly>=0.7.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "ipython>=7.0.0",
    ],
)
