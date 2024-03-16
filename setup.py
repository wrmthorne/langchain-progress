from setuptools import find_packages, setup
import re


__version__ = '0.0.0.dev0'

_deps = [
    'InstructorEmbedding',
    'langchain_community',
    'ray[tune]',
    'responses',
    'sentence-transformers<=2.2.2',
    'tqdm',
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}

def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


REQUIRED_PKGS = deps_list('langchain_community')

EXTRAS = {
    'ray': deps_list('ray[tune]'),
    'tqdm': deps_list('tqdm'),
    'all': deps_list('ray[tune]', 'tqdm'),
    'dev': deps_list('ray[tune]', 'responses', 'sentence-transformers', 'tqdm', 'InstructorEmbedding'),
}

setup(
    name='lanchain_progress',
    license='MIT License',
    url='https://github.com/wrmthorne/langchain-progress',
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS,
    python_requires='>=3.9',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    version=__version__,
    description='Wrapper for nicely displaying progress bars for langchain embedding components when using multiprocessing or ray',
    keywords='Langchain, progress, ray, wrapper, langchain_community, multiprocessing, tqdm',
    author='William Thorne',
    author_email='wthorne1@sheffield.ac.uk',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ]
)