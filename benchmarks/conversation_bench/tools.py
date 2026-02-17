from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

# 1. End Session
end_session_function = FunctionSchema(
    name="end_session", description="End the current session.", properties={}, required=[]
)

# 2. Submit Dietary Request
submit_dietary_request_function = FunctionSchema(
    name="submit_dietary_request",
    description="Submit a dietary request.",
    properties={
        "name": {
            "type": "string",
            "description": "The name of the person making the request.",
        },
        "dietary_preference": {
            "type": "string",
            "description": "The dietary preference (e.g., vegetarian, gluten-free).",
        },
    },
    required=["name", "dietary_preference"],
)

# 3. Submit Session Suggestion
submit_session_suggestion_function = FunctionSchema(
    name="submit_session_suggestion",
    description="Submit a suggestion for a new session.",
    properties={
        "name": {
            "type": "string",
            "description": "The name of the person making the suggestion.",
        },
        "suggestion_text": {
            "type": "string",
            "description": "The text of the session suggestion.",
        },
    },
    required=["name", "suggestion_text"],
)

# 4. Vote for a Session
vote_for_session_function = FunctionSchema(
    name="vote_for_session",
    description="Vote for an existing session.",
    properties={
        "name": {
            "type": "string",
            "description": "The name of the person voting.",
        },
        "session_id": {
            "type": "string",
            "description": "The Session ID of the session being voted for.",
        },
    },
    required=["name", "session_id"],
)

# 5. Request for Tech Support
request_tech_support_function = FunctionSchema(
    name="request_tech_support",
    description="Request technical support.",
    properties={
        "name": {
            "type": "string",
            "description": "The name of the person requesting support.",
        },
        "issue_description": {
            "type": "string",
            "description": "A description of the technical issue.",
        },
    },
    required=["name", "issue_description"],
)

# ======================================================================
# NEW TOOLS FOR HARD BENCHMARK (6-9)
# ======================================================================

# 6. Register for Session
register_for_session_function = FunctionSchema(
    name="register_for_session",
    description="Register an attendee for a specific session. May return an error if the session is full or the registration is a duplicate.",
    properties={
        "name": {
            "type": "string",
            "description": "The name of the person registering.",
        },
        "session_id": {
            "type": "string",
            "description": "The Session ID of the session to register for.",
        },
    },
    required=["name", "session_id"],
)

# 7. Check Schedule Conflict
check_schedule_conflict_function = FunctionSchema(
    name="check_schedule_conflict",
    description="Check if two sessions overlap in time. Returns whether there is a conflict.",
    properties={
        "session_id_1": {
            "type": "string",
            "description": "The Session ID of the first session.",
        },
        "session_id_2": {
            "type": "string",
            "description": "The Session ID of the second session.",
        },
    },
    required=["session_id_1", "session_id_2"],
)

# 8. Cancel Action
cancel_action_function = FunctionSchema(
    name="cancel_action",
    description="Cancel a previous action (e.g., undo a registration, vote, or dietary request). Specify the action type and relevant details.",
    properties={
        "action_type": {
            "type": "string",
            "description": "The type of action to cancel (e.g., 'registration', 'vote', 'dietary_request', 'session_suggestion', 'tech_support').",
        },
        "name": {
            "type": "string",
            "description": "The name of the person whose action is being canceled.",
        },
        "details": {
            "type": "string",
            "description": "Additional details to identify the specific action (e.g., session_id, dietary preference).",
        },
    },
    required=["action_type", "name"],
)

# 9. Add to Schedule
add_to_schedule_function = FunctionSchema(
    name="add_to_schedule",
    description="Add a session to an attendee's personal schedule.",
    properties={
        "name": {
            "type": "string",
            "description": "The name of the person adding the session.",
        },
        "session_id": {
            "type": "string",
            "description": "The Session ID of the session to add to the schedule.",
        },
    },
    required=["name", "session_id"],
)

# Create a ToolsSchema with all 9 tools
ToolsSchemaForTest = ToolsSchema(
    standard_tools=[
        end_session_function,
        submit_dietary_request_function,
        submit_session_suggestion_function,
        vote_for_session_function,
        request_tech_support_function,
        register_for_session_function,
        check_schedule_conflict_function,
        cancel_action_function,
        add_to_schedule_function,
    ]
)
