from setuptools import setup, find_packages

setup(
    name='Complex_Gen',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
        'numpy~=1.26.0',
        'rdkit~=2023.3.3'
        'ase ~=3.22.1'
    ],
)
