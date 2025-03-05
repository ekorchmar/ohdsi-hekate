class RxConceptError(Exception):
    pass


class RxConceptCreationError(RxConceptError):
    """Occurs when a concept cannot be created due to integrity constraints."""


class ForeignNodeCreationError(RxConceptError):
    """Occurs when a foreign node cannot be created due to integrity constraints."""
