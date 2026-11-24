"""Case base maintenance outside the CBR cycle: reviewing the stored cases and restoring the case base.

Retain stores the solutions of single queries during the cycle,
while maintenance revises the contents of the whole case base to facilitate future reasoning,
an activity of its own with its own triggers and quality measures
(D. B. Leake and D. C. Wilson, Categorizing Case-Base Maintenance: Dimensions and Directions, EWCBR 1998;
D. C. Wilson and D. B. Leake, Maintaining Case-Based Reasoners: Dimensions and Directions,
Computational Intelligence 17(2), 2001).
Like the separation of the application cycle and its maintenance in
M. H. Göker and T. Roth-Berghofer, The Development and Utilization of the Case-Based Help-Desk Support System HOMER,
Engineering Applications of Artificial Intelligence 12(6), 1999,
a run needs no query and is not a step of `cbrkit.cycle`.
It reviews the case base by quality measures and restores it by planned changes,
the review and restore steps of
T. Reinartz, I. Iglezakis and T. Roth-Berghofer, Review and Restore for Case-Base Maintenance,
Computational Intelligence 17(2), 2001.
Its changes are the same `cbrkit.model.Changes` of `Add`, `Update` and `Remove` that retain makes,
applied as one mutation of a `cbrkit.typing.Storage`.

Maintaining:
- `Maintainer`: Reviews the stored cases it is given and plans their changes purely,
  merging or removing duplicates, removing inconsistent cases, and removing cases by competence,
  which `cbrkit.model.replan` applies to a storage.

Reviewing, which `Maintainer.plan` returns as the `reviews` of its changes with a competence threshold:
- `Competence`: The competence of every case after Smyth and Keane (1995),
  its coverage, its reachability, and its category (pivotal, spanning, support, or auxiliary, see `CATEGORIES`),
  with footprint-utility deletion and the consistency of the cases.
- `pairwise` and `duplicates`: The similarities of all pairs of cases as a `Matrix` and the groups of duplicates among them.

Triggers stay with the host after Leake and Wilson: periodic, such as a nightly cron job
(`0 3 * * * cbrkit maintain cars.csv '{}'`), conditional, such as after a number of retained cases, or ad hoc,
and a `dry_run` request plans the changes for a person to review before applying them.

Example:
    >>> from cbrkit.sim.numbers import Linear
    >>> maintainer = Maintainer(measure=Linear(max_distance=10), competence=0.9, remove=["auxiliary", "support"])
    >>> changes = maintainer.plan({0: 1.0, 1: 1.3, 2: 9.0})
    >>> [(a.key, a.category) for a in changes.reviews]
    [(0, 'support'), (1, 'support'), (2, 'pivotal')]
    >>> [(change.op, change.key) for change in changes.items]
    [('remove', 0)]
"""

from .build import Maintainer
from .review import CATEGORIES, Competence, Matrix, duplicates, pairwise

__all__ = ["CATEGORIES", "Competence", "Maintainer", "Matrix", "duplicates", "pairwise"]
