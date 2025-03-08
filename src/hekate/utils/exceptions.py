"""Exceptions used elsewhere in the project."""


class CSVReaderError(Exception):
    """Base exception for CSVReader errors."""


class SchemaError(CSVReaderError):
    """Raised when the provided schema does not match the CSV file."""


class RxConceptError(Exception):
    """Base exception for RxConcept errors."""


class RxConceptCreationError(RxConceptError):
    """Occurs when a concept cannot be created due to integrity constraints."""


class ForeignNodeCreationError(RxConceptError):
    """Occurs when a foreign node cannot be created due to integrity constraints."""


class ForeignAttributeError(ForeignNodeCreationError):
    """Occurs when a foreign node specifies an invalid attribute."""


class ForeignDosageStrengthError(ForeignNodeCreationError):
    """Occurs when a foreign node specifies an invalid dosage strength."""
