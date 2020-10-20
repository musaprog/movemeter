import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="movemeter-jkemppainen", # Replace with your own username
    version="0.0.1",
    author="Joni Kemppainen",
    author_email="jjtkemppainen1@sheffield.ac.uk",
    description="A motion analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jkemppainen/movemeter",
    packages=setuptools.find_packages(),
    packages.extend(['tk_steroids']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3) ",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)
