from setuptools import setup, find_packages

setup(
    name="Phugoid NN",
    version="0.1.0",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "numpy",
    ],
    author="Alec Portelli",
    author_email="alecportelli@icloud.com",
    description="Phugoid Neural Network Library",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yourusername/your_project_name",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
