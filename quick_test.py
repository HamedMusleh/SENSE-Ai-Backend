import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_pipeline.triage.triage_classifier import classify_triage
from ai_pipeline.llm.response_strategy import build_strategy

print("=" * 60)
print("Test 1: High Risk")
print("=" * 60)
triage = classify_triage("بدي اموت")
strategy = build_strategy(triage)
print(f"Label: {triage['predicted_label']}")
print(f"Uses LLM: {strategy.use_llm}")
print(f"Response: {strategy.hard_coded_response}")
print()

print("=" * 60)
print("Test 2: Safe")
print("=" * 60)
triage = classify_triage("اليوم لعبت مع صاحبي وكنت مبسوط")
strategy = build_strategy(triage)
print(f"Label: {triage['predicted_label']}")
print(f"Uses LLM: {strategy.use_llm}")
print()

print("✅ Tests passed!")
