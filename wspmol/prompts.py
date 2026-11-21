"""Prompt templates. Development and test splits use disjoint templates."""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a molecular design assistant. Each request lists candidate molecules with their "
    "properties. Choose the candidate most similar to the user's starting molecule that satisfies "
    "every design rule in force for the user's project. Reply with only the candidate letter, "
    "or NONE if no candidate satisfies the rules."
)

MEMORY_HEADER = "Saved project memory:"

SETUP_TEMPLATES = {
    "dev": [
        "Project {project}. My starting molecule is {start}.",
        "I'm working on project {project}. The starting molecule is {start}.",
    ],
    "test": [
        "This is for project {project}; we start from {start}.",
        "Project {project} here. Starting molecule: {start}.",
    ],
}

QUESTION_TEMPLATES = {
    "dev": [
        "Which candidate is closest to my starting molecule?",
        "Pick the best match to the starting molecule from these:",
    ],
    "test": [
        "Which of these is the closest match to {start}?",
        "From the list below, which molecule should we take forward as closest to {start}?",
    ],
}

FOLLOWUP_TEMPLATES = {
    "dev": ["Next set.", "Here is another list."],
    "test": ["New batch of candidates.", "Another round."],
}

LABELS = ["A", "B", "C", "D", "E"]


def render_candidates(cands, sims, start_name: str, id_mode: bool) -> str:
    """cands: list[Molecule] in display order; sims: similarities to the start."""
    if id_mode:
        lines = ["| Option | ID | MW (Da) | Contains Cl | Similarity to start |",
                 "|---|---|---|---|---|"]
        for lab, m, s in zip(LABELS, cands, sims):
            lines.append(f"| {lab} | {m.display_id} | {m.mw:.2f} | {'yes' if m.has_cl else 'no'} | {s:.2f} |")
        return "\n".join(lines)
    return "\n".join(
        f"{lab}. {m.name} (MW {m.mw:.2f} Da; contains chlorine: {'yes' if m.has_cl else 'no'}; "
        f"similarity to {start_name}: {s:.2f})"
        for lab, m, s in zip(LABELS, cands, sims)
    )


def memory_system_prompt(project: str, rule_text: str | None) -> str:
    if rule_text is None:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\n{MEMORY_HEADER}\n- Project {project}: {rule_text}"
