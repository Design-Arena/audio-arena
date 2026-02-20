from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

# 1. End Session
end_session_function = FunctionSchema(
    name="end_session", description="End the current session.", properties={}, required=[]
)

# 2. Check Availability
check_availability_function = FunctionSchema(
    name="check_availability",
    description="Check available appointment slots for a given date and optionally a specific doctor or time preference.",
    properties={
        "date": {
            "type": "string",
            "description": "The date to check availability for (e.g., '2025-01-14').",
        },
        "doctor": {
            "type": "string",
            "description": "The doctor's last name to filter by (e.g., 'Perry'). If omitted, returns slots for all available doctors.",
        },
        "time_preference": {
            "type": "string",
            "description": "Filter by 'morning' (before noon) or 'afternoon' (noon and after). If omitted, returns all slots.",
        },
    },
    required=["date"],
)

# 3. Book Appointment
book_appointment_function = FunctionSchema(
    name="book_appointment",
    description="Book a dental appointment. All required fields must be collected from the patient before calling.",
    properties={
        "patient_name": {
            "type": "string",
            "description": "The full name of the patient.",
        },
        "phone": {
            "type": "string",
            "description": "The patient's phone number.",
        },
        "date": {
            "type": "string",
            "description": "The appointment date (e.g., '2025-01-16').",
        },
        "time": {
            "type": "string",
            "description": "The appointment time in 24-hour format (e.g., '14:15').",
        },
        "doctor": {
            "type": "string",
            "description": "The doctor's last name (e.g., 'Perry').",
        },
        "service_type": {
            "type": "string",
            "description": "The type of service (e.g., 'cleaning', 'checkup', 'filling').",
        },
    },
    required=["patient_name", "phone", "date", "time", "doctor", "service_type"],
)

# 4. Update Patient Info
update_patient_info_function = FunctionSchema(
    name="update_patient_info",
    description="Update a field on an existing appointment (e.g., correct patient name spelling or phone number).",
    properties={
        "appointment_id": {
            "type": "string",
            "description": "The appointment ID to update.",
        },
        "field": {
            "type": "string",
            "description": "The field to update (e.g., 'patient_name', 'phone').",
        },
        "new_value": {
            "type": "string",
            "description": "The corrected value for the field.",
        },
    },
    required=["appointment_id", "field", "new_value"],
)

ToolsSchemaForTest = ToolsSchema(
    standard_tools=[
        end_session_function,
        check_availability_function,
        book_appointment_function,
        update_patient_info_function,
    ]
)
