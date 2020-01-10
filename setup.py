from setuptools import setup, find_packages

setup(
    name='keras_rewiring',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/pabogdan/keras_rewiring',
    license="GNU GPLv3.0",
    author='Petrut Antoniu Bogdan',
    author_email='petrut.bogdan@manchester.ac.uk',
    description='Experiments with rewiring in Keras-defined DNNs',
    # Requirements
    dependency_links=[],

    install_requires=["numpy",
                      "scipy",
                      "keras==2.2.4",
                      "matplotlib",
                      "tensorflow==1.13.1",
                      "argparse",
                      "pillow"],

    classifiers=[
        "Intended Audience :: Science/Research",

        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",

        "Programming Language :: Python :: 3"
        "Programming Language :: Python :: 3.6"
        
        "Topic :: Scientific/Engineering",
    ]
)
