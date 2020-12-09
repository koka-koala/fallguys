from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'gcsfs==0.6.0',
    'pandas==0.24.2',
    'scikit-learn==0.20.4',
    'google-cloud-storage==1.26.0',
    'joblib==0.14.1',
    'numpy==1.18.4',
    'tensorflow-cpu==2.3.1',
    ]

setup(
    name='fallguys',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='fall guys package'
)
