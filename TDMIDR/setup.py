from setuptools import setup, find_packages

setup(
    name='tdmidr',
    version='0.1',
    description='TDMIDR model',
    author='LW',
    author_email='。。。。',
    packages=find_packages(),
    install_requires=[
        'torch',
        'tensorly',
        'numpy',
        'sparsemax',
        'matplotlib',
        'scipy',
        'scikit-learn'
    ],
    python_requires='>=3.8',
)
