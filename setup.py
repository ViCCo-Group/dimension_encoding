from setuptools import setup, find_packages

setup(
    author="Oliver Contier",
    name="dimension_encoding",
    version="0.0.1",
    packages=find_packages(where="dimension_encoding"),
    package_dir={"": "dimension_encoding"},
    # package_dir={":": "src"},
)
