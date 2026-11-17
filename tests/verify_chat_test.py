import os
import sys
from unittest.mock import MagicMock
from streamlit.testing.v1 import AppTest

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set UTF-8 encoding
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Mock answer_question in src.rag_chain to return fixed strings without calling Gemini API
import src.rag_chain

def mock_answer_question(**kwargs):
    q = kwargs.get("question", "")
    return f"Mocked answer for: {q}"

src.rag_chain.answer_question = MagicMock(side_effect=mock_answer_question)

def run_test():
    app_path = os.path.join(PROJECT_ROOT, "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()

    file_bytes = b"%PDF-1.4 test document content"
    file_id = f"test_contract.pdf_{len(file_bytes)}"

    # 1. Upload file
    at.file_uploader[0].upload("test_contract.pdf", file_bytes, "application/pdf").run()

    # 2. Set active document state (mocked analysis ready)
    at.session_state.active_file_id = file_id
    at.session_state.processed_file_id = file_id
    at.session_state.processed_file_name = "test_contract.pdf"
    at.session_state.analysis_complete = True
    at.session_state.vector_store = MagicMock()
    at.session_state.doc_info = {
        "doc_type": "Rental Agreement",
        "category": "rental_lease",
        "parties": ["Landlord", "Tenant"]
    }
    at.session_state.summary = "Test summary content."
    at.session_state.risky_clauses = []
    at.session_state.messages = []

    at.run()

    assert len(at.tabs) == 3, f"Expected 3 tabs, got {len(at.tabs)}"
    assert len(at.chat_input) == 1, f"Expected 1 chat input, got {len(at.chat_input)}"
    assert len(at.chat_message) == 0, f"Expected 0 messages initially, got {len(at.chat_message)}"
    print("[PASS] Initial state initialized. Tab 3 and ChatInput present.")

    # 3. Question 1
    q1 = "What is the monthly rent amount?"
    print(f"\n--- Submitting Question 1: '{q1}' ---")
    at.chat_input[0].set_value(q1).run()

    print(f"Messages count after Q1: {len(at.chat_message)}")
    for i, m in enumerate(at.chat_message):
        print(f"  Message {i+1} [{m.name}]: {m.markdown[0].value}")

    assert len(at.chat_message) == 2, f"Expected 2 messages, got {len(at.chat_message)}"
    assert at.chat_message[0].name == "user" and at.chat_message[0].markdown[0].value == q1
    assert at.chat_message[1].name == "assistant" and "Mocked answer for: " in at.chat_message[1].markdown[0].value

    # 4. Question 2
    q2 = "What is the security deposit?"
    print(f"\n--- Submitting Question 2: '{q2}' ---")
    at.chat_input[0].set_value(q2).run()

    print(f"Messages count after Q2: {len(at.chat_message)}")
    for i, m in enumerate(at.chat_message):
        print(f"  Message {i+1} [{m.name}]: {m.markdown[0].value}")

    assert len(at.chat_message) == 4, f"Expected 4 messages, got {len(at.chat_message)}"
    assert at.chat_message[2].name == "user" and at.chat_message[2].markdown[0].value == q2
    assert at.chat_message[3].name == "assistant" and "Mocked answer for: " in at.chat_message[3].markdown[0].value

    # 5. Question 3
    q3 = "Who pays for electricity and maintenance?"
    print(f"\n--- Submitting Question 3: '{q3}' ---")
    at.chat_input[0].set_value(q3).run()

    print(f"Messages count after Q3: {len(at.chat_message)}")
    for i, m in enumerate(at.chat_message):
        print(f"  Message {i+1} [{m.name}]: {m.markdown[0].value}")

    assert len(at.chat_message) == 6, f"Expected 6 messages, got {len(at.chat_message)}"
    assert at.chat_message[4].name == "user" and at.chat_message[4].markdown[0].value == q3
    assert at.chat_message[5].name == "assistant" and "Mocked answer for: " in at.chat_message[5].markdown[0].value

    # 6. Verify tab structure and check that chat_input is last
    tab3 = at.tabs[2]
    print("\n--- Inspecting Tab 3 children order ---")
    for child_idx, child in tab3.children.items():
        print(f"  Child {child_idx}: {type(child).__name__}")

    last_child_key = max(tab3.children.keys())
    last_child = tab3.children[last_child_key]
    print(f"\nLast child in Tab 3: {type(last_child).__name__}")
    assert type(last_child).__name__ == "ChatInput", f"Expected last element in Tab 3 to be ChatInput, but got {type(last_child).__name__}"

    # Also test friendly error handling on API exception
    print("\n--- Testing Friendly Error Handling ---")
    src.rag_chain.answer_question = MagicMock(side_effect=Exception("Simulated connection timeout"))
    q4 = "Can I keep pets in the apartment?"
    at.chat_input[0].set_value(q4).run()
    print(f"Messages count after Q4 (error scenario): {len(at.chat_message)}")
    for i, m in enumerate(at.chat_message):
        print(f"  Message {i+1} [{m.name}]: {m.markdown[0].value}")
    assert len(at.chat_message) == 8, f"Expected 8 messages, got {len(at.chat_message)}"
    assert at.chat_message[6].name == "user" and at.chat_message[6].markdown[0].value == q4
    assert at.chat_message[7].name == "assistant" and "temporary issue" in at.chat_message[7].markdown[0].value

    # Verify that the last element in tab3 is STILL ChatInput
    tab3 = at.tabs[2]
    last_child_key = max(tab3.children.keys())
    last_child = tab3.children[last_child_key]
    assert type(last_child).__name__ == "ChatInput", f"Expected last element in Tab 3 to be ChatInput, but got {type(last_child).__name__}"

    print("\n[SUCCESS] All verification assertions passed!")

if __name__ == "__main__":
    run_test()
