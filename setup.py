from setuptools import find_packages, setup


__version__ = '0.0.0.dev0'

REQUIRED_PKGS = [
    'langchain_community' # TODO: Add version number here
]

EXTRAS = []

setup(
    name='lanchain_progress',
    license='MIT License',
    classifiers=[],
    url='https://github.com/wrmthorne/langchain-progress',
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED_PKGS,
    extras_required=EXTRAS,
    python_requires='>=3.9', # TODO: See what the lowest version of Python is that we can use
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    version=__version__,
    description='Wrapper for nicely displaying progress bars around langchain components when using ray',
    keywords='Llangchain, progress, ray, wrapper, langchain_community',
    author='William Thorne',
    author_email='wthorne1@sheffield.ac.uk',
)