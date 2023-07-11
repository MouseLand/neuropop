import setuptools

install_deps = ['numpy>=1.20.0', 'scipy', 'natsort',
                'torch>=1.6']

try:
    import torch
    a = torch.ones(2, 3)
    major_version, minor_version, _ = torch.__version__.split(".")
    if major_version == "2" or int(minor_version) >= 6:
        install_deps.remove("torch>=1.6")
except:
    pass

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuropop",
    version="0.1.3",
    author="Marius Pachitariu and Carsen Stringer",
    author_email="stringerc@janelia.hhmi.org",
    description="analysis tools for neural population recordings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MouseLand/rastermap",
    packages=setuptools.find_packages(),
	install_requires = install_deps,
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
)
