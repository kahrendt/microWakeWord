import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="microwakeword",
    version="0.0.1",
    install_requires=[
        "audiomentations",
        "audio_metadata",
        "datasets",
        "mmap_ninja",
        "numpy",
        "pyyaml",
        "tensorflow>=2.14",
    ],
    author="Kevin Ahrendt",
    author_email="kahrendt@gmail.com",
    description="A TensorFlow based wake word detection training framework using synthetic sample generation suitable for certain microcontrollers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kahrendt/microWakeWord",
    project_urls={
        "Bug Tracker": "https://github.com/kahrendt/microWakeWord/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
)
