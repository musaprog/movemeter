import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Version number to __version__ variable
exec(open("movemeter/version.py").read())

install_requires = [
        'numpy',
        'matplotlib',
        'tifffile',
        'Pillow',
        'ExifRead',
        'opencv-python',
        'tk-steroids>=0.3.0',
        ]

setuptools.setup(
    name="movemeter",
    version=__version__,
    author="Joni Kemppainen",
    author_email="jjtkemppainen1@sheffield.ac.uk",
    description="A motion analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jkemppainen/movemeter",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3) ",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)
