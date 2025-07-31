from setuptools import setup, find_packages

setup(
    name="envy-zeitgeist-engine",
    version="0.1.0",
    packages=find_packages(include=["envy_toolkit", "agents", "collectors"]),
    python_requires=">=3.11,<4.0",
)