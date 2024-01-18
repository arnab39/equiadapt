from setuptools import setup, find_packages

setup(
    name='equiadapt',  # Replace with your package's name
    version='0.1.0',  # Package version
    author='Arnab Mondal',  # Replace with your name
    author_email='arnab.mondal@mila.quebec',  # Replace with your email
    description='Library to make any existing neural network architecture equivariant',  # Package summary
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/arnab39/EquivariantAdaptation',  # Replace with your repository URL
    packages=find_packages(where='equiadapt'),
    package_dir={'': 'equiadapt'},
    install_requires=[
        'torch',  # Specify your project's dependencies here
        'numpy',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum version requirement of Python
)