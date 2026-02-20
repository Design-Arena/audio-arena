from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

# 1. End Session
end_session_function = FunctionSchema(
    name="end_session", description="End the current session.", properties={}, required=[]
)

# 2. Search Venues
search_venues_function = FunctionSchema(
    name="search_venues",
    description="Search available venues for a given date, guest count, and optional budget.",
    properties={
        "date": {
            "type": "string",
            "description": "The event date to check availability for (e.g., '2025-03-08').",
        },
        "guest_count": {
            "type": "integer",
            "description": "The expected number of guests.",
        },
        "budget": {
            "type": "number",
            "description": "Optional maximum budget for the venue rental fee.",
        },
    },
    required=["date", "guest_count"],
)

# 3. Check Catering Options
check_catering_options_function = FunctionSchema(
    name="check_catering_options",
    description="Check available catering packages and pricing for a venue and guest count.",
    properties={
        "venue_id": {
            "type": "string",
            "description": "The venue identifier (e.g., 'garden_pavilion').",
        },
        "guest_count": {
            "type": "integer",
            "description": "The number of guests to quote catering for.",
        },
    },
    required=["venue_id", "guest_count"],
)

# 4. Book Event
book_event_function = FunctionSchema(
    name="book_event",
    description="Book an event. All required fields must be collected before calling.",
    properties={
        "name": {
            "type": "string",
            "description": "The name of the person booking the event.",
        },
        "phone": {
            "type": "string",
            "description": "Contact phone number.",
        },
        "date": {
            "type": "string",
            "description": "The event date (e.g., '2025-03-08').",
        },
        "venue_id": {
            "type": "string",
            "description": "The venue identifier (e.g., 'garden_pavilion').",
        },
        "guest_count": {
            "type": "integer",
            "description": "The expected number of guests.",
        },
        "catering_package": {
            "type": "string",
            "description": "The catering package tier (e.g., 'silver', 'gold', 'platinum').",
        },
        "add_ons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of add-on services (e.g., ['dj', 'photographer']).",
        },
    },
    required=["name", "phone", "date", "venue_id", "guest_count", "catering_package", "add_ons"],
)

# 5. Update Event
update_event_function = FunctionSchema(
    name="update_event",
    description="Update a field on an existing event booking.",
    properties={
        "event_id": {
            "type": "string",
            "description": "The event booking ID to update.",
        },
        "field": {
            "type": "string",
            "description": "The field to update (e.g., 'venue_id', 'guest_count', 'catering_package').",
        },
        "new_value": {
            "type": "string",
            "description": "The new value for the field.",
        },
    },
    required=["event_id", "field", "new_value"],
)

# 6. Get Quote
get_quote_function = FunctionSchema(
    name="get_quote",
    description="Get an itemized price quote for an event configuration.",
    properties={
        "venue_id": {
            "type": "string",
            "description": "The venue identifier (e.g., 'garden_pavilion').",
        },
        "guest_count": {
            "type": "integer",
            "description": "The expected number of guests.",
        },
        "catering_package": {
            "type": "string",
            "description": "The catering package tier (e.g., 'silver', 'gold', 'platinum').",
        },
        "add_ons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of add-on services (e.g., ['dj', 'photographer']).",
        },
    },
    required=["venue_id", "guest_count", "catering_package", "add_ons"],
)

ToolsSchemaForTest = ToolsSchema(
    standard_tools=[
        end_session_function,
        search_venues_function,
        check_catering_options_function,
        book_event_function,
        update_event_function,
        get_quote_function,
    ]
)
