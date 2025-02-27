# Hekate

Hekate is an algorithm for automating [International Drug Vocabulary Implementation Process](https://github.com/OHDSI/Vocabulary-v5.0/wiki/International-Drug-Vocabulary-Implementation-Process). Current implementation is a command line tool that takes the described input tables and
[Athena](https://athena.ohdsi.org/) vocabulary download files as input in CSV format and produces the output tables in
CSV format, ready to be processed by the [`GenericUpdate()`](https://github.com/OHDSI/Vocabulary-v5.0/blob/master/working/generic_update.sql) SQL script.

## Technology

Hekate is implemented in Python, requiring version at least `3.13`. Parts of the code will eventually be rewritten to
support Cython for performance reasons.

## Installation
Hekate can be installed using `pip`. Virtual environment is recommended. The package is not yet available on PyPI, so
it must be installed from the repository.

```shell
git clone https://github.com/OHDSI/Hekate.git
cd Hekate
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -e .
```

## Usage
Hekate is (or will be) a command line tool. Eventually, support for a web interface will be added. Even more eventually,
Hekate will be available as a PostgreSQL extension.


## License
Source code is available under MIT License. This means that you can use it for any purpose, including commercial
purposes.

