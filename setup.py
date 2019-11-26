import setuptools

with open("README.md", "r") as f:
	long_description = f.read()

setuptools.setuptools.setup(
    name="phys",
    version="0.0.1",
    author="Ben Warner",
    author_email="warnerbc@plu.edu",
    description="OpenCL-based physics simulation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bcwarner/physics-sim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)