"""Exceptions used elsewhere in the project."""


class CSVReaderError(Exception):
    """
    Base exception for CSVReader errors.
    """


class SchemaError(CSVReaderError):
    """
    Raised when the provided schema does not match the CSV file.
    """


class RxConceptError(Exception):
    """
    Base exception for RxConcept errors.
    """


class InvalidConceptIdError(RxConceptError):
    """
    Occurs when a non-existent or invalid Concept ID is referenced.
    """


class UnmappedSourceConceptError(RxConceptError):
    """
    Occurs when a source concept has no valid mappings to an RxNorm concept.
    """


class RxConceptCreationError(RxConceptError):
    """
    Occurs when a concept cannot be created due to integrity constraints.
    """


class PackCreationError(RxConceptCreationError):
    """
    Occurs when a pack cannot be created due to integrity constraints.
    """


class ForeignNodeCreationError(RxConceptError):
    """
    Occurs when a foreign node cannot be created due to integrity constraints.
    """


class ForeignPackCreationError(PackCreationError, ForeignNodeCreationError):
    """
    Occurs when a source pack node can not be created
    """


class ForeignAttributeError(ForeignNodeCreationError):
    """
    Occurs when a foreign node specifies an invalid attribute.
    """


class ForeignDosageStrengthError(ForeignNodeCreationError):
    """
    Occurs when a foreign node specifies an invalid dosage strength.
    """


class ResolutionError(Exception):
    """
    Base exception for resolution errors, which happen when disambiguating
    between multiple mapping results.
    """
