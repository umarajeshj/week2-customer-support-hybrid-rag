"""20 real-world-style support queries with expected outcomes for ShopEasy.

Each query is labeled with:
  expected: "answer" | "escalate"
  expected_doc_id: the canonical KB entry title or ID that should ground the answer
  must_include: optional substrings that should appear in a correct answer
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected: str  # "answer" or "escalate"
    expected_doc_title: Optional[str]  # Matching your KB metadata titles
    must_include: Tuple[str, ...] = ()


EVAL_CASES: List[EvalCase] = [
    # ---- Should answer: Runbooks & FAQs ----
    EvalCase("How do I reset my password?", "answer", "Account Security and Privacy", ("password",)),
    EvalCase("My payment was declined with error PAY_DECLINED_402, what should I do?", "answer", "Handling Payment Declined Errors (PAY_DECLINED_402)", ("PAY_DECLINED_402",)),
    EvalCase("What is the return window for electronics?", "answer", "Processing Returns and Refunds", ("15-day",)),
    EvalCase("Why is my shipping delayed?", "answer", "Investigating Shipping Delays (SHIP_DELAYED_301)", ("SHIP_DELAYED_301",)),
    EvalCase("Why is promo code SUMMER25OFF not working?", "answer", "Troubleshooting Promo Code Failures (PROMO_INVALID_100)", ("PROMO_INVALID_100",)),
    EvalCase("I received the wrong item in my package, how can I fix this?", "answer", "Handling Wrong Item Received", ("pick error",)),
    EvalCase("My account has been locked with error ACC_LOCKED_503, help!", "answer", "Account Lockout and Suspicious Activity (ACC_LOCKED_503)", ("ACC_LOCKED_503",)),
    EvalCase("An item I ordered shows out of stock after confirmation (INV_SYNC_ERR_700)", "answer", "Resolving Inventory Sync Errors (INV_SYNC_ERR_700)", ("INV_SYNC_ERR_700",)),
    EvalCase("How do my Subscribe & Save recurring orders work?", "answer", "Subscribe & Save Program", ("Subscribe & Save",)),
    EvalCase("What does the ShopEasy Plus membership cover?", "answer", "ShopEasy Plus Membership", ("ShopEasy Plus",)),
    
    # ---- Should answer: Past Tickets & Bugs ----
    EvalCase("My order ORD-88921 shows delivered but I never received it", "answer", "Order ORD-88921 shows delivered but never received", ("FedEx",)),
    EvalCase("Where is my refund for order ORD-91045 after 10 days?", "answer", "Refund not received after 10 business days for ORD-91045", ("Stripe",)),
    EvalCase("My Android app keeps crashing at checkout", "answer", "Mobile app crashes at checkout on Android 14", ("Android 14",)),
    EvalCase("How do I track my order?", "answer", "How do I track my order?", ("tracking number",)),
    EvalCase("What is ShopEasy's return policy?", "answer", "What is ShopEasy's return policy?", ("30 days",)),
    EvalCase("Why was my payment declined?", "answer", "Why was my payment declined?", ("PAY_DECLINED_402",)),
    EvalCase("Can I cancel an order after placing it?", "answer", "Can I cancel an order after placing it?", ("Cancel Order",)),
    EvalCase("How do I contact ShopEasy customer support?", "answer", "How do I contact ShopEasy customer support?", ("Live Chat",)),
    
    # ---- Should escalate: Out-of-scope / Unhandled ----
    EvalCase("Can ShopEasy integrate directly with custom enterprise ERP systems?", "escalate", None),
    EvalCase("Can you help me reset my external bank account login?", "escalate", None),
]