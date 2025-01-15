from pathlib import Path

from setuptools import find_packages
from setuptools import setup

dir_path = Path(__file__).resolve().parent


def read_requirements_file(filename):
    req_file = dir_path.joinpath(filename)
    with req_file.open('r') as f:
        return [line.strip() for line in f]


packages = find_packages(exclude=[])
pkgs = []
for p in packages:
    if p == 'pql' or p.startswith('pql.'):
        pkgs.append(p)

setup(
    name='pql',
    author='Zechu Li',
    license='MIT',
    packages=pkgs,
    install_requires=[
        "hydra-core",
        "loguru",
        "ray",
        "wandb",
        "cloudpickle",
        "scipy",
        "shortuuid",
        "ninja",
    ],
    include_package_data=True,
)