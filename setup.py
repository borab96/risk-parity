from setuptools import setup, find_packages

with open('requirements.txt', 'r', encoding='utf-8') as req:
    req_list = req.read()

setup(
    name='risk-parity-portfolio',
    version='0.0.1',
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    py_modules=['cli', 'rpp'],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'rpp=cli:main'
        ]
    },
    install_requires=[req_list]
)
