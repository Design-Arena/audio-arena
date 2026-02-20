from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

# 1. End Session
end_session_function = FunctionSchema(
    name="end_session", description="End the current session.", properties={}, required=[]
)

# 2. Lookup Item
lookup_item_function = FunctionSchema(
    name="lookup_item",
    description="Search for a product by name, keyword, or item number. Returns matching products with item number, name, size, and price.",
    properties={
        "query": {
            "type": "string",
            "description": "Search query — a product name, keyword, or item number (e.g., 'sourdough loaf', 'maple', '1330').",
        },
    },
    required=["query"],
)

# 3. Process Order
process_order_function = FunctionSchema(
    name="process_order",
    description="Place a new delivery or pickup order. All required fields must be collected from the customer before calling.",
    properties={
        "customer_name": {
            "type": "string",
            "description": "The customer's full name.",
        },
        "phone": {
            "type": "string",
            "description": "The customer's phone number.",
        },
        "items": {
            "type": "array",
            "items": {"type": "object"},
            "description": "List of items to order, each with 'item_id', 'name', 'quantity', and 'unit_price'.",
        },
        "delivery_address": {
            "type": "string",
            "description": "Delivery address. Omit or set to 'pickup' for in-store pickup.",
        },
    },
    required=["customer_name", "phone", "items"],
)

# 4. Update Order
update_order_function = FunctionSchema(
    name="update_order",
    description="Modify an existing order — add, remove, or change quantity of an item.",
    properties={
        "order_id": {
            "type": "string",
            "description": "The order ID to update.",
        },
        "action": {
            "type": "string",
            "description": "The action to perform: 'add', 'remove', or 'change_quantity'.",
        },
        "item_name": {
            "type": "string",
            "description": "The product name.",
        },
        "quantity": {
            "type": "integer",
            "description": "New quantity (for 'add' or 'change_quantity'). Omit for 'remove'.",
        },
    },
    required=["order_id", "action", "item_name"],
)

# 5. Verify Details
verify_details_function = FunctionSchema(
    name="verify_details",
    description="Read back the full order details for customer confirmation, including all items, quantities, prices, and totals.",
    properties={
        "order_id": {
            "type": "string",
            "description": "The order ID to verify.",
        },
    },
    required=["order_id"],
)

ToolsSchemaForTest = ToolsSchema(
    standard_tools=[
        end_session_function,
        lookup_item_function,
        process_order_function,
        update_order_function,
        verify_details_function,
    ]
)
