from setuptools import setup, find_packages

setup(
    name="boxcli",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "click>=8.0.0",
        "numpy<2.0.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "boxcli=cli:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A CLI tool for annotating images with bounding boxes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/boxcli",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 