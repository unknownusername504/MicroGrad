from setuptools import setup

setup(
    name="micrograd",
    version="6.9.420",
    license="MIT",
    packages=[
        "micrograd",
        "micrograd.tensors",
        "micrograd.unittests",
        "micrograd.layers",
        "micrograd.optimizers",
        "micrograd.functions",
    ],
    install_requires=[
        "numpy",
    ],
    python_requires=">=3.8",
)
