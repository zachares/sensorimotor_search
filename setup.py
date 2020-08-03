from setuptools import setup, find_packages

setup(
    name='sens-search',
    packages=[ package for package in find_packages() if package.startswith("perception_learning") or package.startswith("project_utils")],
    description='project specific functions',
    version="1.0",
)

