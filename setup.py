from setuptools import setup  # type: ignore

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
        "micrograd.functions.loss",
        "micrograd.functions.optimizers",
        "micrograd.functions.activations",
        "micrograd.utils",
        "micrograd.scheduler",
    ],
    install_requires=[
        "numpy",
        "psutil",
        "gym",
    ],
    python_requires=">=3.8",
)
