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
