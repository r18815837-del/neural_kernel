from setuptools import find_packages, setup

setup(
    name="neural_kernel",
    version="0.1.0",
    description="Minimal neural kernel with autograd",
    packages=find_packages(),
    install_requires=["numpy>=1.24,<3"],
    python_requires=">=3.9",
)
