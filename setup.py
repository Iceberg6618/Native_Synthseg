from setuptools import find_packages, setup


setup(
    name="Native-Synthseg-Module",
    version="0.1.0",
    description="Run SynthSeg and register segmentations back to native NIfTI space.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="",
    python_requires=">=3.10,<3.11",
    packages=["Native_Synthseg_Module"]
    + find_packages(where="SynthSeg", include=["SynthSeg", "SynthSeg.*", "ext", "ext.*"]),
    package_dir={
        "Native_Synthseg_Module": ".",
        "SynthSeg": "SynthSeg/SynthSeg",
        "ext": "SynthSeg/ext",
    },
    include_package_data=True,
    package_data={
        "Native_Synthseg_Module": [
            "data/models/*.h5",
            "data/labels_classes_priors/*.npy",
            "SynthSeg/data/labels_classes_priors/*.npy",
        ],
    },
    install_requires=[
        "antspyx==0.6.3",
        "h5py==3.8.0",
        "keras==2.10.0",
        "nibabel==5.3.3",
        "numpy==1.23.5",
        "protobuf==3.19.6",
        "scipy==1.10.1",
        "tensorflow-cpu==2.10.1",
    ],
)
