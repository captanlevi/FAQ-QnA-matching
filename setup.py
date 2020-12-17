import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "Bani",
    version = "0.6.2",
    author = "Rushi Babaria",
    url = "https://github.com/captanlevi/FAQ-QnA-matching.git",
    packages = setuptools.find_packages(),
    install_requires = ["sentence_transformers==0.3.8","pandas", "numpy","regex","nlpaug","spacy", "tqdm"],
    author_email="rjlinkin50@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.6'
)

