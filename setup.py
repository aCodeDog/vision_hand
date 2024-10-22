from setuptools import setup, find_packages

setup(
    name='vision_hand',
    version='1.0',
    description='This python package streams diverse tracking data available from AVP to any devices that can communicate with gRPC.',
    author='Younghyo Park',
    author_email='younghyo@mit.edu',
    packages=find_packages(),
    install_requires=[
        'numpy', 'grpcio', 'grpcio-tools', 'matplotlib','isaacgym','avp_stream'
    ],
    extras_require={
    },
)