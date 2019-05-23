from setuptools import setup, find_packages

setup(
    name='dnn_microbench',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/pabogdan/dnn_microbench',
    license="GNU GPLv3.0",
    author='Petrut Antoniu Bogdan',
    author_email='petrut.bogdan@manchester.ac.uk',
    description='Experiments with DNNs to make them amenable for transformation in SNNs',
    # Requirements
    dependency_links=[],

    install_requires=["numpy",
                      "scipy",
                      "keras",
                      "matplotlib"],
    classifiers=[
        "Development Status :: 3 - Alpha",

        "Intended Audience :: Science/Research",

        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",

        "Programming Language :: Python :: 3"
        "Programming Language :: Python :: 3.6"
        
        "Topic :: Scientific/Engineering",
    ]
)
