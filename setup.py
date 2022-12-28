from setuptools import find_packages, setup

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name="dreambooth_tutorial",
    packages=find_packages(),
    version="0.0.1",
    description="Diffusers showcase",
    author="Narek Maloyan",
    license="MIT",
)