"""Cases split into a problem and its solution, see `Case`."""

from dataclasses import dataclass, replace
from typing import Any

__all__ = ["Case", "problem", "solution", "with_solution"]


@dataclass(slots=True, frozen=True, kw_only=True)
class Case[P, R]:
    """A case split into the description of its problem and its solution.

    Cases and queries are always of the same type.
    Any other value is an unsplit case, its own problem and solution, as are the queries of its casebase.
    In a casebase of split cases, a query is a `Case` without a solution,
    and the phases of the cycle compare what fits:
    retrievers and retainers compare problems,
    reusers adapt the case and reuse its solution for the problem of the query,
    revisers assess the whole case for the query,
    and maintainers compare problems and, for the consistency of the cases, solutions.
    Split cases are built by the host,
    while the case base files of `cbrkit.io` and `cbrkit.spec` hold unsplit cases.

    Args:
        problem: Description of the problem.
        solution: The solution of the problem, None for a query.

    Examples:
        >>> from cbrkit.cycle import run_query
        >>> from cbrkit.retrieve import Retriever
        >>> from cbrkit.sim.numbers import Linear
        >>> casebase = {0: Case(problem=10.0, solution="small"), 1: Case(problem=90.0, solution="large")}
        >>> run_query(casebase, Case(problem=80.0), Retriever(measure=Linear(max_distance=100), limit=1)).casebase
        {1: Case(problem=90.0, solution='large')}
    """

    problem: P
    solution: R | None = None


def problem[P, V](case: Case[P, Any] | V, /) -> P | V:
    """The problem of a split case, an unsplit case itself.

    Examples:
        >>> problem(Case(problem="flat tire", solution="patch it")), problem("flat tire")
        ('flat tire', 'flat tire')
    """
    return case.problem if isinstance(case, Case) else case


def solution[R, V](case: Case[Any, R] | V, /) -> R | None | V:
    """The solution of a split case, an unsplit case itself.

    Examples:
        >>> solution(Case(problem="flat tire", solution="patch it")), solution("flat tire")
        ('patch it', 'flat tire')
    """
    return case.solution if isinstance(case, Case) else case


def with_solution(case: Any, solution: Any, /) -> Any:
    """A split case with another solution, the solution itself for an unsplit case, such as in an adaptation.

    Examples:
        >>> with_solution(Case(problem="flat tire", solution="patch it"), "replace it"), with_solution("flat", "fixed")
        (Case(problem='flat tire', solution='replace it'), 'fixed')
    """
    return replace(case, solution=solution) if isinstance(case, Case) else solution
