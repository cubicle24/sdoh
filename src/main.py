from dotenv import load_dotenv
from pathlib import Path
import sys

# Add project root to Python path when running directly
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


from sdoh_agent import run_agent
from pprint import pprint
from sdoh_agent import AgentState

load_dotenv()

def main():
    """Demonstrates the SDOH agent"""

    #load note:
    # note = open("../clinical_notes_repository/note_1_medicine_CHF.txt").read()
    note = open("../clinical_notes_repository/easy_note.txt").read()
    # note = open("../clinical_notes_repository/psychiatric_note.txt").read()
    start_state: AgentState = {
        "note": note,
        "sdoh": {},
        "intervention": {}
    }
    end_state = run_agent(note, start_state)
    pprint(f"Final state: {end_state}")

main()