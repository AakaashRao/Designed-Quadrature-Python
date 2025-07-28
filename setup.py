from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="designed-quadrature",
    version="0.1.0",
    description="Python implementation of Designed Quadrature for maximum simulated likelihood estimation",
    author="Python Port",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.7",
)