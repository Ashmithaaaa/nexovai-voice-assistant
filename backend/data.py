"""
data.py - Mock Clinic Data & Database Management

Manages clinic doctors, available slots, and appointments.
Provides tools for availability checking, booking, and cancellation,
along with event listeners for real-time frontend dashboard synchronization.
"""

import asyncio
import datetime
import uuid
from typing import Any, Callable, Dict, List, Optional

# Async listeners notified on data changes (bookings, cancellations)
_listeners: List[Callable[[], Any]] = []

def register_listener(callback: Callable[[], Any]):
    """Register a callback to be called whenever clinic data changes."""
    if callback not in _listeners:
        _listeners.append(callback)

async def _notify_change(event_type: str, data: Dict[str, Any]):
    """Notify all registered listeners that state has changed."""
    for listener in _listeners:
        try:
            if asyncio.iscoroutinefunction(listener):
                await listener(event_type, data)
            else:
                listener(event_type, data)
        except Exception as e:
            print(f"[Data] Error notifying listener: {e}")

# Initial Doctors & Available Slots
DOCTORS = [
    {
        "id": "doc-1",
        "name": "Dr. Arvind Rao",
        "specialty": "Cardiology",
        "rating": 4.9,
        "avatar": "https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=150&auto=format&fit=crop&q=80",
        "available_slots": [
            {"date": "Tomorrow", "time": "10:00 AM", "status": "available"},
            {"date": "Tomorrow", "time": "02:30 PM", "status": "available"},
            {"date": "Tomorrow", "time": "04:00 PM", "status": "available"},
            {"date": "Thursday", "time": "11:00 AM", "status": "available"},
        ],
    },
    {
        "id": "doc-2",
        "name": "Dr. Priya Sharma",
        "specialty": "Dermatology",
        "rating": 4.8,
        "avatar": "https://images.unsplash.com/photo-1594824813637-44f0c458316f?w=150&auto=format&fit=crop&q=80",
        "available_slots": [
            {"date": "Tomorrow", "time": "11:15 AM", "status": "available"},
            {"date": "Tomorrow", "time": "03:00 PM", "status": "available"},
            {"date": "Thursday", "time": "09:30 AM", "status": "available"},
            {"date": "Thursday", "time": "04:30 PM", "status": "available"},
        ],
    },
    {
        "id": "doc-3",
        "name": "Dr. Rohan Mehta",
        "specialty": "Orthopedics",
        "rating": 4.9,
        "avatar": "https://images.unsplash.com/photo-1537368910025-700350fe46c7?w=150&auto=format&fit=crop&q=80",
        "available_slots": [
            {"date": "Tomorrow", "time": "09:30 AM", "status": "available"},
            {"date": "Tomorrow", "time": "01:00 PM", "status": "available"},
            {"date": "Friday", "time": "11:30 AM", "status": "available"},
        ],
    },
    {
        "id": "doc-4",
        "name": "Dr. Ananya Iyer",
        "specialty": "General Medicine",
        "rating": 4.7,
        "avatar": "https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=150&auto=format&fit=crop&q=80",
        "available_slots": [
            {"date": "Tomorrow", "time": "10:30 AM", "status": "available"},
            {"date": "Tomorrow", "time": "12:00 PM", "status": "available"},
            {"date": "Tomorrow", "time": "05:00 PM", "status": "available"},
            {"date": "Thursday", "time": "02:00 PM", "status": "available"},
        ],
    },
]

# Initial seeded appointments
APPOINTMENTS: List[Dict[str, Any]] = [
    {
        "id": "apt-101",
        "doctor_name": "Dr. Arvind Rao",
        "doctor_specialty": "Cardiology",
        "patient_name": "Siddharth Verma",
        "patient_phone": "9876543210",
        "date": "Tomorrow",
        "time_slot": "11:30 AM",
        "status": "Confirmed",
        "created_at": "Today, 10:15 AM",
    },
    {
        "id": "apt-102",
        "doctor_name": "Dr. Priya Sharma",
        "doctor_specialty": "Dermatology",
        "patient_name": "Kavita Nair",
        "patient_phone": "9123456789",
        "date": "Tomorrow",
        "time_slot": "01:30 PM",
        "status": "Confirmed",
        "created_at": "Today, 11:45 AM",
    },
]


WORD_DIGITS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "oh": "0",
}

def clean_phone(phone_str: str) -> str:
    """Normalize phone number to digits only, supporting both spoken words and numeric strings."""
    if not phone_str:
        return ""
    import re
    tokens = str(phone_str).lower().split()
    converted = []
    for token in tokens:
        cleaned_t = re.sub(r"[^\w]", "", token)
        if cleaned_t in WORD_DIGITS:
            converted.append(WORD_DIGITS[cleaned_t])
        else:
            converted.append(token)
    text = " ".join(converted)
    return "".join(c for c in text if c.isdigit())


def get_doctors_data() -> List[Dict[str, Any]]:
    """Return all doctors and their available slots."""
    return DOCTORS


def get_appointments_data() -> List[Dict[str, Any]]:
    """Return all active appointments."""
    return APPOINTMENTS


def check_doctor_availability(
    doctor_name: Optional[str] = None,
    date: Optional[str] = None,
    specialty: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: check_doctor_availability
    Returns available slots for a doctor, date, or specialty.
    """
    results = []
    doc_query = doctor_name.lower().strip() if doctor_name else ""
    spec_query = specialty.lower().strip() if specialty else ""
    date_query = date.lower().strip() if date else ""

    for doc in DOCTORS:
        # Check doctor name match
        if doc_query:
            # Handle "Dr. Rao", "Rao", "Arvind Rao", etc.
            names = doc["name"].lower().replace("dr.", "").strip().split()
            if not any(q in doc["name"].lower() for q in [doc_query] + doc_query.replace("dr.", "").split()):
                continue

        # Check specialty match
        if spec_query and spec_query not in doc["specialty"].lower():
            continue

        # Filter slots
        available = []
        for slot in doc["available_slots"]:
            if slot["status"] != "available":
                continue
            if date_query and date_query not in slot["date"].lower():
                continue
            available.append(f"{slot['date']} at {slot['time']}")

        if available:
            results.append({
                "doctor": doc["name"],
                "specialty": doc["specialty"],
                "available_slots": available,
            })

    if not results:
        # Return fallback message with general clinic availability
        all_docs = [f"{d['name']} ({d['specialty']})" for d in DOCTORS]
        return {
            "status": "not_found",
            "message": f"No matching available slots found for your criteria. Doctors available at our clinic: {', '.join(all_docs)}.",
            "results": [],
        }

    return {
        "status": "success",
        "count": len(results),
        "results": results,
    }


async def book_appointment(
    patient_phone: str,
    doctor_name: str,
    date: str,
    time_slot: str,
    patient_name: Optional[str] = "Caller",
) -> Dict[str, Any]:
    """
    Tool: book_appointment
    Books a slot for the caller using their spoken phone number.
    Identifies caller ONLY by the spoken phone number.
    """
    clean_p = clean_phone(patient_phone)
    if not clean_p or len(clean_p) < 7:
        return {
            "status": "error",
            "message": "Invalid phone number provided. A valid phone number is required to confirm booking.",
        }

    # Find the doctor
    matched_doc = None
    doc_query = doctor_name.lower().replace("dr.", "").strip()
    for doc in DOCTORS:
        if doc_query in doc["name"].lower():
            matched_doc = doc
            break

    if not matched_doc:
        matched_doc = DOCTORS[0]  # default fallback if approximate

    # Format appointment entry
    apt_id = f"apt-{len(APPOINTMENTS) + 101}"
    new_apt = {
        "id": apt_id,
        "doctor_name": matched_doc["name"],
        "doctor_specialty": matched_doc["specialty"],
        "patient_name": patient_name if patient_name and patient_name.lower() != "caller" else f"Caller ({clean_p[-4:]})",
        "patient_phone": clean_p,
        "date": date or "Tomorrow",
        "time_slot": time_slot,
        "status": "Confirmed",
        "created_at": "Just now",
    }

    # Update slot status in mock doctor slots
    for slot in matched_doc["available_slots"]:
        if (date and date.lower() in slot["date"].lower()) and (time_slot.lower() in slot["time"].lower() or slot["time"].lower() in time_slot.lower()):
            slot["status"] = "booked"
            break
    else:
        # If exact slot wasn't found in predefined list, remove first matching available slot
        for slot in matched_doc["available_slots"]:
            if slot["status"] == "available":
                slot["status"] = "booked"
                break

    APPOINTMENTS.insert(0, new_apt)

    # Broadcast event to frontend dashboard
    await _notify_change("appointment_booked", {
        "appointment": new_apt,
        "doctors": DOCTORS,
        "appointments": APPOINTMENTS,
    })

    return {
        "status": "success",
        "appointment_id": apt_id,
        "doctor": matched_doc["name"],
        "specialty": matched_doc["specialty"],
        "date": new_apt["date"],
        "time_slot": new_apt["time_slot"],
        "patient_phone": clean_p,
        "message": f"Appointment successfully confirmed with {matched_doc['name']} on {new_apt['date']} at {new_apt['time_slot']} for phone number {clean_p}.",
    }


async def cancel_appointment(
    patient_phone: str,
    appointment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool: cancel_appointment
    Cancels existing appointment identified by caller's spoken phone number.
    """
    clean_p = clean_phone(patient_phone)
    if not clean_p:
        return {
            "status": "error",
            "message": "Phone number is required to locate and cancel your appointment.",
        }

    # Search for active appointments with this phone number
    matching = [a for a in APPOINTMENTS if clean_phone(a["patient_phone"]) == clean_p and a["status"] == "Confirmed"]

    if not matching:
        # Also try partial match (last 6 digits or similar) if length is close
        matching = [a for a in APPOINTMENTS if (clean_p in clean_phone(a["patient_phone"]) or clean_phone(a["patient_phone"]) in clean_p) and a["status"] == "Confirmed"]

    if not matching:
        return {
            "status": "not_found",
            "message": f"No active appointment was found registered under the phone number {clean_p}.",
        }

    target_apt = matching[0]
    target_apt["status"] = "Cancelled"

    # Free up slot in doctor's availability
    for doc in DOCTORS:
        if doc["name"] == target_apt["doctor_name"]:
            for slot in doc["available_slots"]:
                if slot["status"] == "booked":
                    slot["status"] = "available"
                    break

    # Broadcast event to frontend dashboard
    await _notify_change("appointment_cancelled", {
        "appointment": target_apt,
        "doctors": DOCTORS,
        "appointments": APPOINTMENTS,
    })

    return {
        "status": "success",
        "appointment_id": target_apt["id"],
        "doctor": target_apt["doctor_name"],
        "date": target_apt["date"],
        "time_slot": target_apt["time_slot"],
        "patient_phone": clean_p,
        "message": f"Appointment with {target_apt['doctor_name']} on {target_apt['date']} at {target_apt['time_slot']} has been successfully cancelled.",
    }
