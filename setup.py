from setuptools import find_packages, setup
import pathlib


HERE = pathlib.Path(__file__).parent
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

setup(
    name="inferdata",
    packages=find_packages(),
    version="0.0.1",
    description="Library for Tabular data type inference fro ML projects",
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author="Daniiar Abdiev",
    author_email="daniiar.abdiev@gmail.com",
    url='https://github.com/daniiarabdiev/inferdata',
    license="MIT",
    python_requires='>=3.6',
    install_requires=['joblib>=0.16.0', 'nltk>=3.5', 'numpy>=1.19.2', 'pandas>=1.1.2',
                      'scikit-learn>=0.23.2', 'scipy>=1.5.2', 'sklearn>=0.0']
)