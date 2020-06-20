import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="detection_3d-Denis-Tananaev",
    version="0.0.1",
    author="Denis Tananaev",
    author_email="d.d.tananaev@gmail.com",
    description="3D bbox detection with Lidar",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dtananaev/lidar_dynamic_objects_detection",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
