"""Two fixed People Desk tasks shared by the chat UI and API clients.

Each task is a different multi-step workflow. The question, employee id, and
tool path are fixed so POST /demos/{id} and POST /chat return the same answer
and the same policy citations on every call.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoTask:
    id: str
    label: str
    blurb: str
    query: str
    employee_id: str
    citation_source: str

    def as_dict(self) -> dict:
        chat = {"query": self.query, "employee_id": self.employee_id}
        return {
            "id": self.id,
            "label": self.label,
            "blurb": self.blurb,
            "query": self.query,
            "employee_id": self.employee_id,
            "citation_source": self.citation_source,
            "chat": chat,
            "curl": f"curl -X POST http://127.0.0.1:8000/demos/{self.id}",
        }


DEMOS: tuple[DemoTask, ...] = (
    DemoTask(
        id="remote-emp-1008",
        label="Demo 1 · Remote eligibility · EMP-1008",
        blurb=(
            "Looks up Alex Kim, reads the hub location rule in POL-RMT-003, "
            "and checks that fully remote work needs a People Ops reclass."
        ),
        query="Am I eligible for fully remote work living in Tacoma?",
        employee_id="EMP-1008",
        citation_source="03-remote-hybrid-work.md",
    ),
    DemoTask(
        id="pto-emp-1014",
        label="Demo 2 · PTO guidance · EMP-1014",
        blurb=(
            "Looks up Devon Walsh, checks the HarborHub PTO balance and the "
            "30-day use rule, and cites eligibility in POL-PTO-001."
        ),
        query="Can I take PTO next week?",
        employee_id="EMP-1014",
        citation_source="01-paid-time-off.md",
    ),
)


def get_demo(demo_id: str) -> DemoTask | None:
    for demo in DEMOS:
        if demo.id == demo_id:
            return demo
    return None
