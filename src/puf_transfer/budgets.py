from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetSplit:
    total: int
    train: int
    validation: int


def split_labeled_budget(
    total: int,
    *,
    validation_fraction: float = 0.2,
    min_validation: int = 20,
    max_validation: int | None = None,
) -> BudgetSplit:
    """Split a labeled-CRP budget without creating hidden validation cost."""
    if total < 2:
        raise ValueError("A labeled budget must contain at least two CRPs.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")

    validation = max(min_validation, int(round(total * validation_fraction)))
    if max_validation is not None:
        validation = min(validation, max_validation)
    validation = min(validation, total - 1)
    return BudgetSplit(total, total - validation, validation)
