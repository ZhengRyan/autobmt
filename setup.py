#!/usr/bin/env python

"""The setup script."""

import re
from setuptools import setup, find_packages

NAME = 'autobmt'

with open('README.md') as readme_file:
    readme = readme_file.read()


def get_version():
    with open(f"{NAME}/__init__.py") as f:
        return re.search(r'\d+\.\d+\.\d+', f.read()).group()


def get_requirements(stage=None):
    file_name = 'requirements'

    if stage is not None:
        file_name = f"{file_name}-{stage}"

    requirements = []
    with open(f"{file_name}.txt", 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('-'):
                continue

            requirements.append(line)

    return requirements


test_requirements = ['pytest', ]

setup(
    author="RyanZheng",
    author_email='zhengruiping000@163.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="a modeling tool that automatically builds scorecards and tree models.",
    install_requires=get_requirements(),
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='autobmt',
    name=NAME,
    packages=find_packages(include=['autobmt', 'autobmt.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ZhengRyan/autobmt',
    version=get_version(),
    zip_safe=False,
)
