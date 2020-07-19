from setuptools import setup, find_packages

required = ["pytest", "coverage", "pytest-cov", "black"]

setup(
    name="lac",
    version="0.0.1",
    description="A companion to study linear algebra",
    url="https://github.com/srcolinas/lac",
    python_requires=">=3.6",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    package_dir={"": "."},
)
