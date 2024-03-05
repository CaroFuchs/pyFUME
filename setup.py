import setuptools

setuptools.setup(
    name = 'pyFUME', 
    version = '0.3.1',
    author = 'Caro Fuchs',
    author_email = 'c.e.m.fuchs@tue.nl',
    description = 'A Python package for fuzzy model estimation',
    long_description = open('README.md').read(),
    long_description_content_type =  'text/markdown',
    keywords = ['fuzzy logic', 'fuzzy inference systems', 'fuzzy model','data-driven', 'model estimation', 'machine learning'],
    url='https://github.com/CaroFuchs/pyFUME',
    packages=setuptools.find_packages(),
    license='LICENSE.txt',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[ 'scipy', 'numpy', 'simpful', 'fst-pso', 'pandas', 'typing_extensions'],
)
