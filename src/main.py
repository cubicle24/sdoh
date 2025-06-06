import os
from dotenv import load_dotenv
from sdoh_agent import run_agent
from pprint import pprint
from sdoh_agent import AgentState


load_dotenv()

def main():
    """Demonstrates the SDOH agent"""

    #load note:
    # note = open("../clinical_notes_repository/note_1_medicine_CHF.txt").read()
    # note = open("../clinical_notes_repository/easy_note.txt").read()
    note = open("../clinical_notes_repository/psychiatric_note.txt").read()
    start_state: AgentState = {
        "note": note,
        "sdoh": {},
        "intervention": {}
    }
    end_state = run_agent(note, start_state)
    pprint(f"Final state: {end_state}")

if __name__ == "__main__":
    main()