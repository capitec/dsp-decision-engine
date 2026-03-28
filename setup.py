from setuptools import setup, find_packages

setup(
    name="ped",
    version="0.0.1",
    packages=find_packages(include=["ped", "ped.*"]),
    python_requires=">=3.8",
    install_requires=[
        "polars",
        "pandas",
    ],
    author="Sholto Armstrong, Christiaan van As",
    description="A basic pipeline execution framework",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
