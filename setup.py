from setuptools import setup, find_packages

setup(
    name="protstruc",
    packages=find_packages(exclude=[]),
    include_package_data=True,
    version="0.0.1",
    license="MIT",
    description="Utility library for handling protein structures in Python.",
    author="Dohoon Lee",
    author_email="dohlee.bioinfo@gmail.com",
    long_description_content_type="text/markdown",
    url="https://github.com/dohlee/protstruc",
    keywords=[
        "bioinformatics",
        "computational biology",
        "artificial intelligence",
    ],
    install_requires=[
        "numpy",
        "biopandas>=0.4.1",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
