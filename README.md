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
>   * Errors in source tables are logged and do not interrupt the run. This behavior will be configurable in the future.
>   * Not all of the described functionality is implemented yet.

To efficiently build an extension of the existing hierarchy, Hekate performs the following steps:
1. Reading the Athena vocabulary files. Integrity checks are performed on input concepts, discarding the concepts that should not be included in the hierarchy as targets for mapping. This logic is implemented in [src/hekate/csv_read/athena.py] module.
2. Reading the source data tables. The source data is expected to be in CSV format. This logic is implemented in [src/hekate/csv_read/source_input.csv] module. Source concepts get represented as a flat set of ForeignNodePrototype instances.
3. Translating source data into native RxNorm representation respecting the precedence of mappings. This is implemented in [src/hekate/rx_model/hierarchy/translator.py] module. For each source ForeignNodePrototype, multiple ForeignNodes are instantiated, from precedence mappings.
4. Traversing the existing hierarchy to find the terminal nodes matching the native definition. This is implemented in [src/hekate/rx_model/hierarchy/traversal.py] module.
5. Picking the best match for each ForeignNodePrototype, ranking the matches by quality of the match and lowest precedence. This is implemented in [src/hekate/process/resolve.py] module.

Currently not implemented, but is being actively worked on:
 * Robust reporting of errors both in RxNorm hierarchy and source data.
 * Shaping output for integration with `GenericUpdate()` script.
 * Automated hierarchy extension to accommodate new concepts as implemented by `BuildRxE.sql`

Currently not implemented, as design decisions are subject of active discussion:
 * Precise Ingredient logic for source data.
 * Handling of currently unused `possible_excipient` field.

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
Hekate is a command line tool. Eventually, support for a web interface will be added. Even more eventually,
Hekate will be available as a PostgreSQL extension.

After installing with `pip`, you can run the tool with the following command:

```shell
hekate --help
```

## License
Source code is available under MIT License. This means that you can use it for any purpose, including commercial purposes.
