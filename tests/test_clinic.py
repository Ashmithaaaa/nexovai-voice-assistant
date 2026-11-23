"""
tests/test_clinic.py - Automated Unit Tests for Clinic Tools
"""

import asyncio
from backend.data import (
    check_doctor_availability,
    book_appointment,
    cancel_appointment,
    get_doctors_data,
    get_appointments_data,
    clean_phone,
)

def test_clean_phone():
    assert clean_phone("9876543210") == "9876543210"
    assert clean_phone("+91 (987) 654-3210") == "919876543210"
    assert clean_phone("invalid") == ""

def test_check_availability():
    res = check_doctor_availability(doctor_name="Rao")
    assert res["status"] == "success"
    assert len(res["results"]) > 0
    assert "Dr. Arvind Rao" in res["results"][0]["doctor"]

def test_booking_and_cancellation():
    async def _async_test():
        test_phone = "9988776655"
        res_book = await book_appointment(
            patient_phone=test_phone,
            doctor_name="Dr. Arvind Rao",
            date="Tomorrow",
            time_slot="10:00 AM",
            patient_name="Test Patient",
        )
        assert res_book["status"] == "success"
        assert res_book["patient_phone"] == test_phone

        # Verify cancellation
        res_cancel = await cancel_appointment(patient_phone=test_phone)
        assert res_cancel["status"] == "success"
        assert res_cancel["doctor"] == "Dr. Arvind Rao"

    asyncio.run(_async_test())
