from setuptools import setup, find_packages

setup(
    name="rlExperiments",
    version="0.1",
    description="Deep Reinforcement Learning Experiments",
    author="Eric janto",
    url="https://github.com/ericjanto/deep-rl-experiments",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    install_requires=[
        "numpy>=1.18",
        "torch>=1.3",
        "gymnasium>=0.26",
        "gymnasium[box2d]",
        "tqdm>=4.41",
        "pyglet>=1.3",
        "matplotlib>=3.1",
        "pytest>=5.3",
        "pytest-csv>=3.0",
        "pytest-json>=0.4",
        "pytest-json-report>=1.5",
        "pytest-timeout>=2.1",
        "highway-env"
    ],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
