import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().split('\n')


setuptools.setup(
    name='deepsysid',
    version='0.0.1',
    author='Alexandra Baier',
    author_email='alexandra.baier@ipvs.uni-stuttgart.de',
    description='System identification toolkit for multistep prediction using deep learning and hybrid methods',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
