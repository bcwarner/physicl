import setuptools

with open("README.md", "r") as f:
	long_description = f.read()

setuptools.setuptools.setup(
    name="physicl",
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
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Framework :: Pytest",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha"
    ],
    python_requires='>=3.6',
)