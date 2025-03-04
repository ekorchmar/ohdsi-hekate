# Hekate

Hekate is an algorithm for automating [International Drug Vocabulary Implementation Process](https://github.com/OHDSI/Vocabulary-v5.0/wiki/International-Drug-Vocabulary-Implementation-Process). Current implementation is a command line tool that takes the described input tables and
[Athena](https://athena.ohdsi.org/) vocabulary download files as input in CSV format and produces the output tables in
CSV format, ready to be processed by the [`GenericUpdate()`](https://github.com/OHDSI/Vocabulary-v5.0/blob/master/working/generic_update.sql) SQL script.

## Technology

Hekate is implemented in Python, requiring version at least `3.13`. Parts of the code will eventually be rewritten to
support Cython for performance reasons.

### Libraries
Hekate uses the following libraries:
  * [Polars](https://pola.rs/) for tabular data manipulation. Main reason for not using Pandas is that Polars is much
     more performant on Athena vocabulary files, able do discard unnecessary columns and rows while reading the file.
  * [rustworkx](https://www.rustworkx.org/) for graph manipulation, to build a traversable representation of the
    RxNorm and RxNorm Extension hierarchies.


### Mode of operation

> [!WARNING]
> This section documents design decisions that may change in the future.
>   * Filtering source concepts will produce additional artifacts for QA purposes.
>   * Data filtering will be decoupled from reading to allow for operation on an SQL backend.
>   * Encountering errors in the input tables will raise an exception and stop the process. This may not be the default
>     behavior in the future.
>   * Not all of the described functionality is implemented yet.

To efficiently build an extension of the existing hierarchy, Hekate performs the following steps:
1. Reading the input tables and Athena vocabulary files. Integrity checks are performed on input concepts, discarding
  the concepts that should not be included in the hierarchy as targets for mapping. This logic is implemented in
  [src/hekate/athena.py] module.
2. Building a graph representation of the RxNorm and RxNorm Extension as data is being done as data is read. This is
  implemented in [src/hekate/rx_model/hierarchy.py] module.
3. Reading the input tables from the input CSV files. This is implemented in [src/hekate/csv_read/input.py] module
  (TBD). Note that any errors in the input tables will raise an exception and stop the process.
4. Traversing the existing hierarchy and automatically extending to accomodate the new concepts. This is implemented in
  [src/hekate/rx_model/extension.py] module (TBD).
5. Exporting the output tables to CSV files. This is implemented in [src/hekate/csv_write/output.py] module (TBD). The
  format of the output is stage tables for the `GenericUpdate()` script.

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

