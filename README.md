# Hekate

Hekate is an algorithm for automating [International Drug Vocabulary Implementation Process](https://github.com/OHDSI/Vocabulary-v5.0/wiki/International-Drug-Vocabulary-Implementation-Process). Current implementation is a command line tool that takes the described input tables and
[Athena](https://athena.ohdsi.org/) vocabulary download files as input in CSV format and produces the output tables in
CSV format, ready to be processed by the [`GenericUpdate()`](https://github.com/OHDSI/Vocabulary-v5.0/blob/master/working/generic_update.sql) SQL script.

## Technology

Hekate is implemented in Python, requiring version at least `3.10`. Although it is possible to run the script by the
Python interpreter, it is intended to be compiled into binary using [Cython](https://cython.org/). To compile Hekate for
your platform, run the following command:

```shell
# TODO: Add the command to compile the script
```
