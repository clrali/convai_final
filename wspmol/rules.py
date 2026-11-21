"""Design constraints (project rules) and their revisions."""
from __future__ import annotations

from dataclasses import dataclass

from .molecules import Molecule


@dataclass(frozen=True)
class Rule:
    kind: str                 # "mass" | "chlorine" | "none"
    mass_limit: float = 250.0

    def allows(self, m: Molecule) -> bool:
        if self.kind == "mass":
            return m.mw <= self.mass_limit
        if self.kind == "chlorine":
            return not m.has_cl
        return True

    def text(self) -> str:
        if self.kind == "mass":
            return f"Only consider molecules with a molecular weight of at most {self.mass_limit:g} daltons."
        if self.kind == "chlorine":
            return "Exclude molecules containing chlorine."
        return "No constraints."

    def short(self) -> str:
        if self.kind == "mass":
            return f"molecular weight <= {self.mass_limit:g} Da"
        if self.kind == "chlorine":
            return "no chlorine"
        return "none"


NO_RULE = Rule("none")
RULES = {"mass": Rule("mass", 250.0), "chlorine": Rule("chlorine")}
REVISED_MASS_LIMIT = 350.0


def revised(rule: Rule) -> Rule:
    """C4 revision: raise the mass limit, or allow chlorine."""
    if rule.kind == "mass":
        return Rule("mass", REVISED_MASS_LIMIT)
    if rule.kind == "chlorine":
        return NO_RULE
    return rule


def revision_text(rule: Rule, split: str, rng) -> str:
    if rule.kind == "mass":
        opts = {
            "dev": [f"Update: raise the molecular weight limit to {REVISED_MASS_LIMIT:g} daltons.",
                    f"Change of plan: the mass limit is now {REVISED_MASS_LIMIT:g} Da."],
            "test": [f"We can relax the mass rule. Molecules up to {REVISED_MASS_LIMIT:g} daltons are fine now.",
                     f"Please use a new molecular weight limit of {REVISED_MASS_LIMIT:g} daltons from here on."],
        }[split]
    else:
        opts = {
            "dev": ["Update: chlorine is now allowed.",
                    "Change of plan: you may include chlorinated molecules."],
            "test": ["We can drop the chlorine restriction. Chlorine-containing molecules are fine now.",
                     "Please allow molecules with chlorine from here on."],
        }[split]
    return rng.choice(opts)
