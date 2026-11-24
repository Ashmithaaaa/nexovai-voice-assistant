import asyncio
from backend.bot import handle_simulated_dialogue, GREETING

def test_dialogue_repetition_and_clarification():
    async def run():
        events = []
        async def dummy_callback(evt):
            events.append(evt)

        state = {
            "pending_action": None,
            "pending_doc": "Dr. Arvind Rao",
            "pending_date": "Tomorrow",
            "pending_slot": "10:00 AM",
            "pending_phone": None,
            "last_agent_prompt": GREETING,
        }

        # 1. Unclear input should ask to repeat, NOT say "glad to book"
        reply = await handle_simulated_dialogue("uh hello what", state, dummy_callback)
        assert "glad to book" not in reply.lower()
        assert ("catch" in reply.lower() or "help" in reply.lower())

        # 2. Asking for availability
        reply = await handle_simulated_dialogue("Does Dr. Rao have anything tomorrow?", state, dummy_callback)
        assert "slots available" in reply.lower()
        assert "glad to book" not in reply.lower()

        # 3. Asking to repeat
        reply = await handle_simulated_dialogue("could you please repeat that?", state, dummy_callback)
        assert "certainly" in reply.lower()

        # 4. Explicit booking intent
        reply = await handle_simulated_dialogue("I want to book tomorrow at 10 AM", state, dummy_callback)
        assert "glad to book" in reply.lower()
        assert state["pending_action"] == "book"
        assert state["pending_phone"] is None

        # 5. When waiting for phone number, if caller mumbles or says something unclear:
        # MUST NOT repeat "glad to book"! It should ask for the phone number!
        reply = await handle_simulated_dialogue("wait what was that", state, dummy_callback)
        assert "glad to book" not in reply.lower()
        assert "phone number" in reply.lower()

        # 6. Caller speaks phone number
        reply = await handle_simulated_dialogue("my number is 9876543210", state, dummy_callback)
        assert "9 8 7 6 5 4 3 2 1 0" in reply
        assert "correct" in reply.lower()
        assert state["pending_phone"] == "9876543210"

        # 7. When waiting for confirmation, if caller says something unclear:
        # MUST NOT say "glad to book"! It should ask to confirm yes or no!
        reply = await handle_simulated_dialogue("sorry can you hear me", state, dummy_callback)
        assert "glad to book" not in reply.lower()
        assert "confirm" in reply.lower()

        # 8. Caller confirms with "yes"
        reply = await handle_simulated_dialogue("yes, that's correct", state, dummy_callback)
        assert "confirmed" in reply.lower()
        assert state["pending_action"] is None
        assert state["pending_phone"] is None

    asyncio.run(run())


def test_specialty_matching_dermatology_and_others():
    async def run():
        events = []
        async def dummy_callback(evt):
            events.append(evt)

        # Case 1: "I want to consult an dematrology" -> MUST return Dr. Priya Sharma / Dermatology, NOT Dr. Arvind Rao!
        state1 = {
            "pending_action": None,
            "pending_doc": None,
            "pending_specialty": None,
            "pending_date": "Tomorrow",
            "pending_slot": None,
            "pending_phone": None,
            "last_agent_prompt": GREETING,
        }
        reply1 = await handle_simulated_dialogue("i want to consult an dematrology", state1, dummy_callback)
        assert "dr. priya sharma" in reply1.lower()
        assert "dermatology" in reply1.lower()
        assert "dr. arvind rao" not in reply1.lower()
        assert "cardiology" not in reply1.lower()

        # Case 2: Booking dermatology
        reply2 = await handle_simulated_dialogue("I want to book tomorrow at 11:15 AM", state1, dummy_callback)
        assert "dr. priya sharma" in reply2.lower()
        assert "11:15 am" in reply2.lower()
        assert "phone number" in reply2.lower()

        # Case 3: Asking about Orthopedics / joint pain
        state2 = {
            "pending_action": None,
            "pending_doc": None,
            "pending_specialty": None,
            "pending_date": "Tomorrow",
            "pending_slot": None,
            "pending_phone": None,
            "last_agent_prompt": GREETING,
        }
        reply3 = await handle_simulated_dialogue("can i see an orthopedic doctor for knee pain?", state2, dummy_callback)
        assert "dr. rohan mehta" in reply3.lower()
        assert "orthopedics" in reply3.lower()

        # Case 4: Asking about General Medicine / fever
        state3 = {
            "pending_action": None,
            "pending_doc": None,
            "pending_specialty": None,
            "pending_date": "Tomorrow",
            "pending_slot": None,
            "pending_phone": None,
            "last_agent_prompt": GREETING,
        }
        reply4 = await handle_simulated_dialogue("i have a fever and need a general physician", state3, dummy_callback)
        assert "dr. ananya iyer" in reply4.lower()
        assert "general medicine" in reply4.lower()

        # Case 5: Direct slot booking with phone number and confirmation for Dermatology
        state4 = {
            "pending_action": None,
            "pending_doc": None,
            "pending_specialty": None,
            "pending_date": "Tomorrow",
            "pending_slot": None,
            "pending_phone": None,
            "last_agent_prompt": GREETING,
        }
        reply5a = await handle_simulated_dialogue("I'd like to book Dr. Priya Sharma for Tomorrow at 11:15 AM. My phone number is 9876543210.", state4, dummy_callback)
        assert "9 8 7 6 5 4 3 2 1 0" in reply5a
        assert "correct" in reply5a.lower()
        assert state4["pending_doc"] == "Dr. Priya Sharma"

        reply5b = await handle_simulated_dialogue("yes, that's right", state4, dummy_callback)
        assert "confirmed" in reply5b.lower()
        assert "dr. priya sharma" in reply5b.lower()

        # Case 6: Exact user scenario: "Yes book I want Thursday at 9:30 a.m."
        # MUST NOT repeat the available slots! Must proceed to ask for phone number!
        state5 = {
            "pending_action": None,
            "pending_doc": "Dr. Priya Sharma",
            "pending_specialty": "Dermatology",
            "pending_date": "Tomorrow",
            "pending_slot": None,
            "pending_phone": None,
            "last_agent_prompt": "Dr. Priya Sharma in Dermatology has the following slots available: Thursday at 09:30 AM, Thursday at 04:30 PM. Would you like to book one of these?",
        }
        reply6a = await handle_simulated_dialogue("Yes book I want Thursday at 9:30 a.m.", state5, dummy_callback)
        assert "glad to book" in reply6a.lower()
        assert "dr. priya sharma" in reply6a.lower()
        assert "09:30 am" in reply6a.lower()
        assert "thursday" in reply6a.lower()
        assert "phone number" in reply6a.lower()
        assert "would you like to book one of these" not in reply6a.lower()

        # Case 7: Cancellation flow with phone confirmation
        state6 = {
            "pending_action": None,
            "pending_doc": "Dr. Priya Sharma",
            "pending_specialty": "Dermatology",
            "pending_date": "Tomorrow",
            "pending_slot": None,
            "pending_phone": None,
            "last_agent_prompt": GREETING,
        }
        reply7a = await handle_simulated_dialogue("I want to cancel my appointment", state6, dummy_callback)
        assert "cancel" in reply7a.lower()
        assert "phone number" in reply7a.lower()

        reply7b = await handle_simulated_dialogue("my phone is 9876543210", state6, dummy_callback)
        assert "cancel" in reply7b.lower()
        assert "booking" not in reply7b.lower() # Must say cancel, NOT booking!
        assert "9 8 7 6 5 4 3 2 1 0" in reply7b

        # Case 8: Abort / "never mind"
        state7 = {
            "pending_action": "book",
            "pending_doc": "Dr. Arvind Rao",
            "pending_specialty": "Cardiology",
            "pending_date": "Tomorrow",
            "pending_slot": "10:00 AM",
            "pending_phone": None,
            "last_agent_prompt": "What is your phone number?",
        }
        reply8 = await handle_simulated_dialogue("never mind forget it", state7, dummy_callback)
        assert "cleared" in reply8.lower()
        assert state7["pending_action"] is None

        # Case 9: Gratitude & Farewell
        state8 = {
            "pending_action": None,
            "pending_doc": None,
            "pending_specialty": None,
            "pending_date": "Tomorrow",
            "pending_slot": None,
            "pending_phone": None,
            "last_agent_prompt": GREETING,
        }
        reply9 = await handle_simulated_dialogue("Thank you so much, bye!", state8, dummy_callback)
        assert "welcome" in reply9.lower()

    asyncio.run(run())

