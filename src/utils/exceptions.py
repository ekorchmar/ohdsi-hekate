"""Exceptions used elsewhere in the project."""


class CSVReaderError(Exception):
    """Base exception for CSVReader errors."""


class SchemaError(CSVReaderError):
    """Raised when the provided schema does not match the CSV file."""
