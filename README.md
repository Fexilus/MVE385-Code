# MVE385-code
The code for the Viscando project 4 (Enhanced object tracking) 2020 as part of MVE385 at Chalmers University of Technology.

## Project structure
The code is separated into two parts.

The actual implementation is formatted as a package located in `/tracking`.
The `pyproject.toml`, `setup.cgf` and `setup.py` files specify the dependencies and other information about the package.
These dependencies should be kept to a minimum, as they will be required for all use cases of the code.
The name of this package should probably be changed, and possibly be put into a relevant namespace (`viscando.*`?).

The presentation of the computations is formatted as separate scripts located in `/examples`.
The dependencies of these scripts are kept in `requirements.txt`.
This includes a dependency on the package, loaded in edit mode.
Thus the scripts serve a secondary function as a testing environment during development.
As the scripts are only necessary when presenting and developing the code, these dependencies do not have to be as neatly kept.

## Usage
To use only the package containing the computational code, install it using a build tool such as `setuptools` or `build`.
To use the scripts, install the dependencies in `requirements.txt` using your preferred package management method, such as one of the following commands:

    $ pip install -r requirements.txt
    $ pipenv install
    $ conda install --file requirements.txt
