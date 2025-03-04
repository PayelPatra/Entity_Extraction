from setuptools import setup, find_packages

setup(
    name="medical_entity_extraction",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "pandas",
        "nltk",
        "sklearn",
        "tqdm",
    ],
)
