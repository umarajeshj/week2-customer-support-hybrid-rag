"""Run the 20-query evaluation suite and report First-Contact Resolution (FCR)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List
from eval_set import EVAL_CASES, EvalCase


@dataclass
class RowResult:
    case: EvalCase
    bot_decision: str
    top_doc_title: str
    answer: str
    correct: bool
    failure_mode: str  # "" | "wrong_doc" | "missing_phrase" | "false_answer" | "false_escalate"


def evaluate_pipeline(ensemble_retriever, llm, prompt_template) -> List[RowResult]:
    rows: List[RowResult] = []
    
    for case in EVAL_CASES:
        # 1. Retrieve docs via your hybrid ensemble
        retrieved_docs = ensemble_retriever.invoke(case.query) if ensemble_retriever else []
        top_doc_title = retrieved_docs[0].metadata.get("title", "") if retrieved_docs else ""
        
        # 2. Determine bot action (answer vs escalate based on confidence/context availability)
        decision = "answer" if retrieved_docs else "escalate"
        
        # 3. Generate answer if answering
        answer = ""
        if decision == "answer":
            context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
            prompt_val = prompt_template.format(context=context_str, question=case.query)
            answer = llm.invoke(prompt_val).content

        # 4. Grade the response
        correct = False
        failure_mode = ""

        if case.expected == "escalate":
            if decision == "escalate":
                correct = True
            else:
                failure_mode = "false_answer"
        else:  # answer expected
            if decision == "escalate":
                failure_mode = "false_escalate"
            elif top_doc_title.lower() != case.expected_doc_title.lower():
                failure_mode = f"wrong_doc (got: {top_doc_title})"
            else:
                missing = [p for p in case.must_include if p.lower() not in answer.lower()]
                if missing:
                    failure_mode = f"missing_phrase: {missing}"
                else:
                    correct = True

        rows.append(RowResult(
            case=case,
            bot_decision=decision,
            top_doc_title=top_doc_title,
            answer=answer,
            correct=correct,
            failure_mode=failure_mode,
        ))
        
    return rows