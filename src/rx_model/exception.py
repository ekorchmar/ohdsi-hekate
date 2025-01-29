class RxConceptError(Exception):
    pass


class RxConceptCreationError(RxConceptError):
    """Occurs when a concept cannot be created due to integrity constraints."""
    pass


class StrengthUnitMismatchError(RxConceptError):
    """Occurs when different strength units are used in subtyping drugs."""

    def __init__(self, msg, unit1, unit2):
        super().__init__(msg)
        self.unit1 = unit1
        self.unit2 = unit2
