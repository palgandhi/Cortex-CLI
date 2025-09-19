from setuptools import setup, find_packages

setup(
    name='cortex-cli',
    version='0.1.0',
    packages=find_packages(where='.', exclude=['tests']),  # The key change is here
    entry_points={
        'console_scripts': [
            'cortex = cortex.main:main',
        ],
    },
    install_requires=[
        'pandas',
        'spacy',
        'fuzzywuzzy',
        'python-Levenshtein',
        'scikit-learn',
         'torch', # New dependency
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A powerful CLI for machine learning.',
    # long_description=open('README.md').read(),  # Temporarily comment out this line
    # long_description_content_type='text/markdown', # and this one
    url='https://github.com/your-username/cortex',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)