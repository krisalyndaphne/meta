from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, conlist


class Invoice(BaseModel):
    invoice_id: str
    vendor_name: str
    amount: float
    tax_code: str
    due_date: str
    po_number: str
    currency: str
    line_items: List[str]
    received_at: str


class PurchaseOrder(BaseModel):
    po_number: str
    vendor_name: str
    approved_amount: float
    allowed_tax_codes: List[str]
    approval_threshold: float


class VendorProfile(BaseModel):
    vendor_name: str
    risk_level: Literal["low", "medium", "high"]
    known_aliases: List[str]
    historical_flags: List[str]


class PolicyRules(BaseModel):
    max_due_days: int
    duplicate_window_days: int
    split_threshold_amount: float


class Observation(BaseModel):
    task_id: str
    invoice_id: str
    vendor_name: str
    amount: float
    tax_code: str
    due_date: str
    po_number: str
    risk_level: str
    allowed_actions: List[str]
    audit_trail: List[str]
    context: Dict[str, str]


class Action(BaseModel):
    action_type: Literal[
        "review_invoice",
        "request_correction",
        "approve_invoice",
        "escalate_case",
        "ask_vendor_question",
    ]
    payload: Dict[str, str | List[str]]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class Reward(BaseModel):
    value: float
    breakdown: Dict[str, float]


class TaskFixture(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    invoices: conlist(Invoice, min_length=1)
    purchase_orders: List[PurchaseOrder]
    vendors: List[VendorProfile]
    policies: PolicyRules
    expected_outcome: str
    perfect_actions: List[str]
    worst_actions: List[str]
    max_steps: int
    clean_invoice_ids: List[str]
    escalation_required_invoice_ids: List[str]
    contradiction_pairs: List[List[str]]
    success_keywords: List[str]
    unsafe_approval_keywords: List[str]
    expected_checks: Optional[List[str]] = None
