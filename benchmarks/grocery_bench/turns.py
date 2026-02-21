"""
Grocery benchmark turns (30 turns: 0-29).

Scenario: Carmen Reyes calls Harvest & Hearth Market to place a grocery
delivery order. 

Scoring dimensions:
  Core (every turn):
  - KB:        KB / factual grounding
  - Instr:     Instruction following
  - Tool:      Tool-use correctness
  Conditional (added per-category):
  - State:     State tracking across turns
  - Ambiguity: Ambiguity handling (entities, numbers, context)
"""

# ============================================================================
# TURNS 0-7: ORDER SETUP + FIRST ITEMS
# ============================================================================

turns = [
# Turn 0 — Opening: customer requests a delivery order
{'input': "Hi, I'd like to place an order for delivery please.",
 'golden_text': "Of course! I'd be happy to help you place a delivery order. "
                "Can I get your name, delivery address, and phone number to "
                "get started?",
 'required_function_call': None,
 'function_call_response': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_000.wav'},

# Turn 1 — Name spelling + delivery address + phone number
{'input': "My name is Carmen Reyes — that's C-A-R-M-E-N Carmen, "
          "R-E-Y-E-S Reyes. Delivery to 3840 Willow Lane, Pasadena. "
          "Phone is 818-940-3617.",
 'golden_text': "Got it — Carmen Reyes, C-A-R-M-E-N, R-E-Y-E-S. "
                "Delivering to 3840 Willow Lane, Pasadena. Phone number "
                "8-1-8, 9-4-0, 3-6-1-7. Is all that correct?",
 'required_function_call': None,
 'function_call_response': None,
 'categories': ['basic_qa', 'numerical_reasoning'],
 'audio_file': 'audio/turn_001.wav'},

# Turn 2 — First item: flour (homophone trap: flour vs flower)
{'input': "Yes. I need flour for baking. Five pound bag.",
 'golden_text': "All-Purpose Flour, five-pound bag for six ninety-nine. "
                "Added to your order.",
 'required_function_call': {'name': 'lookup_item',
                            'args': {'query': 'flour'}},
 'function_call_response': {'status': 'success',
                            'results': [{'item_id': '1010',
                                         'name': 'All-Purpose Flour',
                                         'size': '5 lb bag',
                                         'price': 6.99}]},
 'categories': ['tool_use', 'ambiguous_entity'],
 'audio_file': 'audio/turn_002.wav'},

# Turn 3 — Vague pronoun: "make that two" → model must infer flour
{'input': "Actually, make that two.",
 'golden_text': "Updated to two bags of All-Purpose Flour — thirteen "
                "ninety-eight for the flour.",
 'required_function_call': None,
 'function_call_response': None,
 'categories': ['long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_003.wav'},

# Turn 4 — Mid-sentence self-correction: rolls → loaf
{'input': "I need the sourdough rolls— actually, the loaf.",
 'golden_text': "Sourdough Loaf, seven fifty for one loaf. Got it — the "
                "loaf, not the rolls. Adding that.",
 'required_function_call': {'name': 'lookup_item',
                            'args': {'query': 'sourdough loaf'}},
 'function_call_response': {'status': 'success',
                            'results': [{'item_id': '5010',
                                         'name': 'Sourdough Loaf',
                                         'size': '1 loaf',
                                         'price': 7.50}]},
 'categories': ['tool_use', 'ambiguous_entity'],
 'audio_file': 'audio/turn_004.wav'},

# Turn 5 — Straightforward item lookup: organic eggs
{'input': "And some organic eggs.",
 'golden_text': "Organic Eggs, one dozen for six forty-nine. Adding.",
 'required_function_call': {'name': 'lookup_item',
                            'args': {'query': 'organic eggs'}},
 'function_call_response': {'status': 'success',
                            'results': [{'item_id': '2015',
                                         'name': 'Organic Eggs',
                                         'size': '1 dozen',
                                         'price': 6.49}]},
 'categories': ['tool_use', 'ambiguous_entity'],
 'audio_file': 'audio/turn_005.wav'},

# Turn 6 — Relative quantity math: "double the eggs" (1→2 dozen)
{'input': "Double the eggs.",
 'golden_text': "Updated to two dozen Organic Eggs — twelve ninety-eight "
                "for the eggs.",
 'required_function_call': None,
 'function_call_response': None,
 'categories': ['long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_006.wav'},

# Turn 7 — Straightforward item lookup: gray sea salt
{'input': "Oh, and gray sea salt — the eight-ounce jar.",
 'golden_text': "Gray Sea Salt, eight-ounce jar for eight forty-nine. "
                "Added.",
 'required_function_call': {'name': 'lookup_item',
                            'args': {'query': 'gray sea salt'}},
 'function_call_response': {'status': 'success',
                            'results': [{'item_id': '3010',
                                         'name': 'Gray Sea Salt',
                                         'size': '8 oz jar',
                                         'price': 8.49}]},
 'categories': ['tool_use'],
 'audio_file': 'audio/turn_007.wav'},

# ============================================================================
# TURNS 8-16: MORE ITEMS + DISAMBIGUATION CHALLENGES
# ============================================================================

# Turn 8 — Off-topic KB question mid-order: delivery timing
{'input': "By the way — is same-day delivery available?",
 'golden_text': "If you place your order before two PM, you're eligible "
                "for same-day delivery, subject to availability. The "
                "standard delivery window is two to four hours from "
                "order confirmation.",
 'required_function_call': None,
 'function_call_response': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_008.wav'},

# Turn 9 — Maple syrup (similar name to maple candy — disambiguation later)
{'input': "OK good. Now I need maple syrup.",
 'golden_text': "Maple Syrup, Grade A, twelve-ounce bottle for nine "
                "ninety-nine. Added.",
 'required_function_call': {'name': 'lookup_item',
                            'args': {'query': 'maple syrup'}},
 'function_call_response': {'status': 'success',
                            'results': [{'item_id': '1330',
                                         'name': 'Maple Syrup (Grade A)',
                                         'size': '12 oz bottle',
                                         'price': 9.99}]},
 'categories': ['tool_use', 'ambiguous_entity'],
 'audio_file': 'audio/turn_009.wav'},

# Turn 10 — Quantity update: maple syrup 1→2 bottles
{'input': "Actually, make that two bottles of the maple syrup.",
 'golden_text': "Updated to two bottles of Maple Syrup — nineteen "
                "ninety-eight for the syrup.",
 'required_function_call': None,
 'function_call_response': None,
 'categories': ['long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_010.wav'},

# Turn 11 — Maple candy (must not confuse with maple syrup already ordered)
{'input': "And also the maple candy.",
 'golden_text': "Maple Candy, six-ounce box for thirteen ninety-nine. "
                "Added.",
 'required_function_call': {'name': 'lookup_item',
                            'args': {'query': 'maple candy'}},
 'function_call_response': {'status': 'success',
                            'results': [{'item_id': '1303',
                                         'name': 'Maple Candy',
                                         'size': '6 oz box',
                                         'price': 13.99}]},
 'categories': ['tool_use', 'ambiguous_entity'],
 'audio_file': 'audio/turn_011.wav'},

# Turn 12 — False memory trap: user confuses syrup/candy prices
{'input': "And the maple syrup was thirteen dollars ninety-nine, right?",
 'golden_text': "No — the Maple Syrup is nine ninety-nine per bottle. "
                "You have two bottles for nineteen ninety-eight. Thirteen "
                "ninety-nine is the Maple Candy. Different items, "
                "different prices.",
 'required_function_call': None,
 'function_call_response': None,
 'categories': ['long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_012.wav'},

# Turn 13 — Audio false start: "four—" sounds like quantity, actually "for the"
{'input': "I need four— for the— I mean, add olive oil.",
 'golden_text': "Extra Virgin Olive Oil, sixteen-ounce bottle for twelve "
                "ninety-nine. Added.",
 'required_function_call': {'name': 'lookup_item',
                            'args': {'query': 'olive oil'}},
 'function_call_response': {'status': 'success',
                            'results': [{'item_id': '4015',
                                         'name': 'Extra Virgin Olive Oil',
                                         'size': '16 oz bottle',
                                         'price': 12.99}]},
 'categories': ['tool_use', 'ambiguous_entity'],
 'audio_file': 'audio/turn_013.wav'},

# Turn 14 — Partial name: "the olives" is a separate product from olive oil
{'input': "Add the olives too.",
 'golden_text': "Kalamata Olives, ten-ounce jar for eight forty-nine. "
                "Added — that's separate from the olive oil.",
 'required_function_call': {'name': 'lookup_item',
                            'args': {'query': 'olives'}},
 'function_call_response': {'status': 'success',
                            'results': [{'item_id': '4051',
                                         'name': 'Kalamata Olives',
                                         'size': '10 oz jar',
                                         'price': 8.49}]},
 'categories': ['tool_use', 'ambiguous_entity'],
 'audio_file': 'audio/turn_014.wav'},

# Turn 15 — Ambiguous "both Italian ones": lookup returns two, add both
{'input': "Italian parsley... actually, I'll take both Italian ones.",
 'golden_text': "I found two Italian items — Italian Parsley, one bunch "
                "for two forty-nine, and Italian Seasoning, three-ounce "
                "jar for five ninety-nine. Adding both to your order.",
 'required_function_call': {'name': 'lookup_item',
                            'args': {'query': 'italian'}},
 'function_call_response': {'status': 'success',
                            'results': [{'item_id': '6020',
                                         'name': 'Italian Parsley',
                                         'size': '1 bunch',
                                         'price': 2.49},
                                        {'item_id': '3020',
                                         'name': 'Italian Seasoning',
                                         'size': '3 oz jar',
                                         'price': 5.99}]},
 'categories': ['tool_use', 'ambiguous_entity'],
 'audio_file': 'audio/turn_015.wav'},

# Turn 16 — Three items in one utterance: honey, balsamic, vanilla
{'input': "Also throw in some honey, balsamic vinegar, and vanilla extract.",
 'golden_text': "Raw Honey, twelve-ounce jar for ten ninety-nine. "
                "Balsamic Vinegar, eight-ounce bottle for nine ninety-nine. "
                "Vanilla Extract, four-ounce bottle for eleven ninety-nine. "
                "All three added.",
 'required_function_call': [
     {'name': 'lookup_item', 'args': {'query': 'honey'}},
     {'name': 'lookup_item', 'args': {'query': 'balsamic vinegar'}},
     {'name': 'lookup_item', 'args': {'query': 'vanilla extract'}}],
 'function_call_response': [
     {'status': 'success',
      'results': [{'item_id': '4030',
                   'name': 'Raw Honey',
                   'size': '12 oz jar',
                   'price': 10.99}]},
     {'status': 'success',
      'results': [{'item_id': '4020',
                   'name': 'Balsamic Vinegar',
                   'size': '8 oz bottle',
                   'price': 9.99}]},
     {'status': 'success',
      'results': [{'item_id': '1035',
                   'name': 'Vanilla Extract (pure)',
                   'size': '4 oz bottle',
                   'price': 11.99}]}],
 'categories': ['tool_use'],
 'audio_file': 'audio/turn_016.wav'},

# ============================================================================
# TURNS 17-20: HOMOPHONE, CONDITIONAL LOGIC, CHAINED EDITS, SUBTOTAL
# ============================================================================

# Turn 17 — Homophone collision: "flower bouquet" alongside baking "flour"
{'input': "And a flower bouquet for my wife.",
 'golden_text': "Fresh Flower Bouquet, mixed seasonal for thirty-four "
                "ninety-nine. Added — and just to be clear, that's the "
                "floral bouquet, separate from the baking flour on your "
                "order.",
 'required_function_call': {'name': 'lookup_item',
                            'args': {'query': 'flower bouquet'}},
 'function_call_response': {'status': 'success',
                            'results': [{'item_id': '8010',
                                         'name': 'Fresh Flower Bouquet',
                                         'size': 'mixed seasonal',
                                         'price': 34.99}]},
 'categories': ['tool_use', 'ambiguous_entity'],
 'audio_file': 'audio/turn_017.wav'},

# Turn 18 — Conditional addition: "under seven dollars? If so, add it"
{'input': "Do you have whole milk under seven dollars? If so, add it.",
 'golden_text': "Organic Whole Milk, one gallon for six ninety-nine — "
                "that's under seven dollars, so I'm adding it.",
 'required_function_call': {'name': 'lookup_item',
                            'args': {'query': 'whole milk'}},
 'function_call_response': {'status': 'success',
                            'results': [{'item_id': '2010',
                                         'name': 'Organic Whole Milk',
                                         'size': '1 gallon',
                                         'price': 6.99}]},
 'categories': ['tool_use', 'numerical_reasoning'],
 'audio_file': 'audio/turn_018.wav'},

# Turn 19 — Chained edits in one utterance: syrup 2→1, parsley 1→2
{'input': "Change the syrup back to one bottle and double the parsley.",
 'golden_text': "Updated — Maple Syrup down to one bottle for nine "
                "ninety-nine, and Italian Parsley doubled to two bunches "
                "for four ninety-eight. Your running subtotal is now "
                "one hundred seventy-four thirty-three.",
 'required_function_call': None,
 'function_call_response': None,
 'categories': ['long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_019.wav'},

# Turn 20 — Fifteen/fifty audio confusion + self-correction + subtotal request
{'input': "Oh, and organic bananas — fifty pounds. "
          "Wait, no, fifteen. One-five. What's my running total?",
 'golden_text': "Organic Bananas, ninety-nine cents per pound. Fifteen "
                "pounds comes to fourteen eighty-five. Added. Your running "
                "subtotal is now one hundred eighty-nine eighteen.",
 'required_function_call': {'name': 'lookup_item',
                            'args': {'query': 'organic bananas'}},
 'function_call_response': {'status': 'success',
                            'results': [{'item_id': '6015',
                                         'name': 'Organic Bananas',
                                         'size': 'per lb',
                                         'price': 0.99}]},
 'categories': ['tool_use', 'ambiguous_entity', 'numerical_reasoning'],
 'audio_file': 'audio/turn_020.wav'},

# ============================================================================
# TURNS 21-22: PHONE CORRECTION, PLACE ORDER, READ-BACK
# ============================================================================

# Turn 21 — Phone correction (3617→3670) + place order (16 items)
{'input': "My number is actually 818-940-3670, not 3617. "
          "OK, go ahead and place it.",
 'golden_text': "Got it — updated phone to 8-1-8, 9-4-0, 3-6-7-0. "
                "Order placed! Your order number is ORD-5521. Subtotal "
                "is one hundred eighty-nine eighteen, and your "
                "order qualifies for free delivery in Zone 1. "
                "Total: one hundred eighty-nine eighteen.",
 'required_function_call': {'name': 'process_order',
                            'args': {
                                'customer_name': 'Carmen Reyes',
                                'phone': '818-940-3670',
                                'items': [
                                    {'item_id': '1010', 'name': 'All-Purpose Flour',
                                     'quantity': 2, 'unit_price': 6.99},
                                    {'item_id': '5010', 'name': 'Sourdough Loaf',
                                     'quantity': 1, 'unit_price': 7.50},
                                    {'item_id': '2015', 'name': 'Organic Eggs',
                                     'quantity': 2, 'unit_price': 6.49},
                                    {'item_id': '3010', 'name': 'Gray Sea Salt',
                                     'quantity': 1, 'unit_price': 8.49},
                                    {'item_id': '1330', 'name': 'Maple Syrup (Grade A)',
                                     'quantity': 1, 'unit_price': 9.99},
                                    {'item_id': '1303', 'name': 'Maple Candy',
                                     'quantity': 1, 'unit_price': 13.99},
                                    {'item_id': '4015', 'name': 'Extra Virgin Olive Oil',
                                     'quantity': 1, 'unit_price': 12.99},
                                    {'item_id': '4051', 'name': 'Kalamata Olives',
                                     'quantity': 1, 'unit_price': 8.49},
                                    {'item_id': '6020', 'name': 'Italian Parsley',
                                     'quantity': 2, 'unit_price': 2.49},
                                    {'item_id': '3020', 'name': 'Italian Seasoning',
                                     'quantity': 1, 'unit_price': 5.99},
                                    {'item_id': '4030', 'name': 'Raw Honey',
                                     'quantity': 1, 'unit_price': 10.99},
                                    {'item_id': '4020', 'name': 'Balsamic Vinegar',
                                     'quantity': 1, 'unit_price': 9.99},
                                    {'item_id': '1035', 'name': 'Vanilla Extract (pure)',
                                     'quantity': 1, 'unit_price': 11.99},
                                    {'item_id': '8010', 'name': 'Fresh Flower Bouquet',
                                     'quantity': 1, 'unit_price': 34.99},
                                    {'item_id': '2010', 'name': 'Organic Whole Milk',
                                     'quantity': 1, 'unit_price': 6.99},
                                    {'item_id': '6015', 'name': 'Organic Bananas',
                                     'quantity': 15, 'unit_price': 0.99}],
                                'delivery_address': '3840 Willow Lane, Pasadena'}},
 'function_call_response': {'status': 'success',
                            'order_id': 'ORD-5521',
                            'subtotal': 189.18,
                            'delivery_fee': 0.00,
                            'total': 189.18},
 'categories': ['tool_use', 'long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_021.wav'},

# Turn 22 — Full order read-back via verify_details (16 items)
{'input': "Wait, read me each item before I go.",
 'golden_text': "Here's your order for Carmen Reyes, delivery to 3840 "
                "Willow Lane, Pasadena, phone 818-940-3670:\n\n"
                "1. All-Purpose Flour, 5 lb × 2 — $13.98\n"
                "2. Sourdough Loaf — $7.50\n"
                "3. Organic Eggs, 1 dozen × 2 — $12.98\n"
                "4. Gray Sea Salt, 8 oz — $8.49\n"
                "5. Maple Syrup, 12 oz — $9.99\n"
                "6. Maple Candy, 6 oz — $13.99\n"
                "7. Extra Virgin Olive Oil, 16 oz — $12.99\n"
                "8. Kalamata Olives, 10 oz — $8.49\n"
                "9. Italian Parsley × 2 — $4.98\n"
                "10. Italian Seasoning, 3 oz — $5.99\n"
                "11. Raw Honey, 12 oz — $10.99\n"
                "12. Balsamic Vinegar, 8 oz — $9.99\n"
                "13. Vanilla Extract, 4 oz — $11.99\n"
                "14. Fresh Flower Bouquet — $34.99\n"
                "15. Organic Whole Milk, 1 gal — $6.99\n"
                "16. Organic Bananas, 15 lb — $14.85\n\n"
                "Subtotal: $189.18. Free delivery. Total: $189.18.",
 'required_function_call': {'name': 'verify_details',
                            'args': {'order_id': 'ORD-5521'}},
 'function_call_response': {'status': 'success',
                            'order': {
                                'order_id': 'ORD-5521',
                                'customer': 'Carmen Reyes',
                                'phone': '818-940-3670',
                                'delivery_address': '3840 Willow Lane, Pasadena',
                                'items': [
                                    {'item_id': '1010', 'name': 'All-Purpose Flour',
                                     'qty': 2, 'price': 6.99},
                                    {'item_id': '5010', 'name': 'Sourdough Loaf',
                                     'qty': 1, 'price': 7.50},
                                    {'item_id': '2015', 'name': 'Organic Eggs',
                                     'qty': 2, 'price': 6.49},
                                    {'item_id': '3010', 'name': 'Gray Sea Salt',
                                     'qty': 1, 'price': 8.49},
                                    {'item_id': '1330', 'name': 'Maple Syrup (Grade A)',
                                     'qty': 1, 'price': 9.99},
                                    {'item_id': '1303', 'name': 'Maple Candy',
                                     'qty': 1, 'price': 13.99},
                                    {'item_id': '4015', 'name': 'Extra Virgin Olive Oil',
                                     'qty': 1, 'price': 12.99},
                                    {'item_id': '4051', 'name': 'Kalamata Olives',
                                     'qty': 1, 'price': 8.49},
                                    {'item_id': '6020', 'name': 'Italian Parsley',
                                     'qty': 2, 'price': 2.49},
                                    {'item_id': '3020', 'name': 'Italian Seasoning',
                                     'qty': 1, 'price': 5.99},
                                    {'item_id': '4030', 'name': 'Raw Honey',
                                     'qty': 1, 'price': 10.99},
                                    {'item_id': '4020', 'name': 'Balsamic Vinegar',
                                     'qty': 1, 'price': 9.99},
                                    {'item_id': '1035', 'name': 'Vanilla Extract (pure)',
                                     'qty': 1, 'price': 11.99},
                                    {'item_id': '8010', 'name': 'Fresh Flower Bouquet',
                                     'qty': 1, 'price': 34.99},
                                    {'item_id': '2010', 'name': 'Organic Whole Milk',
                                     'qty': 1, 'price': 6.99},
                                    {'item_id': '6015', 'name': 'Organic Bananas',
                                     'qty': 15, 'price': 0.99}],
                                'subtotal': 189.18,
                                'delivery_fee': 0.00,
                                'total': 189.18}},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_022.wav'},

# ============================================================================
# TURNS 23-29: POST-ORDER MODIFICATIONS + FINAL RECONCILIATION
# ============================================================================

# Turn 23 — Compound edit: remove candy + swap balsamic for Dijon mustard
{'input': "Take off the maple candy. And swap the balsamic vinegar "
          "for Dijon mustard.",
 'golden_text': "Removed the Maple Candy. And I've swapped out the "
                "Balsamic Vinegar for Dijon Mustard, eight-ounce jar "
                "for four forty-nine. Your subtotal is now one hundred "
                "sixty-nine sixty-nine.",
 'required_function_call': [
     {'name': 'update_order',
      'args': {'order_id': 'ORD-5521',
               'action': 'remove',
               'item_name': 'Maple Candy'}},
     {'name': 'update_order',
      'args': {'order_id': 'ORD-5521',
               'action': 'remove',
               'item_name': 'Balsamic Vinegar'}},
     {'name': 'lookup_item',
      'args': {'query': 'dijon mustard'}},
     {'name': 'update_order',
      'args': {'order_id': 'ORD-5521',
               'action': 'add',
               'item_name': 'Dijon Mustard',
               'quantity': 1}}],
 'function_call_response': [
     {'status': 'success'},
     {'status': 'success'},
     {'status': 'success',
      'results': [{'item_id': '4025',
                   'name': 'Dijon Mustard',
                   'size': '8 oz jar',
                   'price': 4.49}]},
     {'status': 'success'}],
 'categories': ['tool_use', 'long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_023.wav'},

# Turn 24 — Retroactive quantity change: flour 2→1
{'input': "And go back to just one bag of flour.",
 'golden_text': "Updated to one bag of All-Purpose Flour. Your subtotal "
                "is now one hundred sixty-two seventy.",
 'required_function_call': {'name': 'update_order',
                            'args': {'order_id': 'ORD-5521',
                                     'action': 'change_quantity',
                                     'item_name': 'All-Purpose Flour',
                                     'quantity': 1}},
 'function_call_response': {'status': 'success'},
 'categories': ['tool_use', 'long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_024.wav'},

# Turn 25 — Revert earlier removal: add balsamic vinegar back
{'input': "Actually, put the balsamic vinegar back on too.",
 'golden_text': "Balsamic Vinegar added back — eight-ounce bottle for "
                "nine ninety-nine. Your subtotal is now one hundred "
                "seventy-two sixty-nine.",
 'required_function_call': {'name': 'update_order',
                            'args': {'order_id': 'ORD-5521',
                                     'action': 'add',
                                     'item_name': 'Balsamic Vinegar',
                                     'quantity': 1}},
 'function_call_response': {'status': 'success'},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_025.wav'},

# Turn 26 — Conditional removal by price: "anything over ten dollars"
{'input': "Actually, take off anything over ten dollars.",
 'golden_text': "Let me check your order — items over ten dollars per unit "
                "are Extra Virgin Olive Oil at twelve ninety-nine, Raw Honey "
                "at ten ninety-nine, Vanilla Extract at eleven ninety-nine, "
                "and the Fresh Flower Bouquet at thirty-four ninety-nine. "
                "Removing all four. Your subtotal is now one hundred one "
                "seventy-three.",
 'required_function_call': [
     {'name': 'update_order',
      'args': {'order_id': 'ORD-5521',
               'action': 'remove',
               'item_name': 'Extra Virgin Olive Oil'}},
     {'name': 'update_order',
      'args': {'order_id': 'ORD-5521',
               'action': 'remove',
               'item_name': 'Raw Honey'}},
     {'name': 'update_order',
      'args': {'order_id': 'ORD-5521',
               'action': 'remove',
               'item_name': 'Vanilla Extract'}},
     {'name': 'update_order',
      'args': {'order_id': 'ORD-5521',
               'action': 'remove',
               'item_name': 'Fresh Flower Bouquet'}}],
 'function_call_response': [
     {'status': 'success'},
     {'status': 'success'},
     {'status': 'success'},
     {'status': 'success'}],
 'categories': ['tool_use', 'long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_026.wav'},

# Turn 27 — Implicit reference: "whatever I ordered first" + subtotal + delivery check
{'input': "Add another one of whatever I ordered first. "
          "What's my total now? Is delivery still free?",
 'golden_text': "Your first item was All-Purpose Flour — I've bumped it "
                "back to two bags for thirteen ninety-eight. Your new "
                "subtotal is one hundred eight seventy-two. That's still "
                "over the seventy-five dollar threshold, so delivery is "
                "still free.",
 'required_function_call': {'name': 'update_order',
                            'args': {'order_id': 'ORD-5521',
                                     'action': 'change_quantity',
                                     'item_name': 'All-Purpose Flour',
                                     'quantity': 2}},
 'function_call_response': {'status': 'success'},
 'categories': ['tool_use', 'long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_027.wav'},

# Turn 28 — Final read-back via verify_details (12 items after all edits)
{'input': "Read me the final order one more time.",
 'golden_text': "Here's your final order:\n\n"
                "Customer: Carmen Reyes\n"
                "Phone: 818-940-3670\n"
                "Delivery: 3840 Willow Lane, Pasadena\n\n"
                "1. All-Purpose Flour, 5 lb × 2 — $13.98\n"
                "2. Sourdough Loaf — $7.50\n"
                "3. Organic Eggs, 1 dozen × 2 — $12.98\n"
                "4. Gray Sea Salt, 8 oz — $8.49\n"
                "5. Maple Syrup, 12 oz — $9.99\n"
                "6. Kalamata Olives, 10 oz — $8.49\n"
                "7. Italian Parsley × 2 — $4.98\n"
                "8. Italian Seasoning, 3 oz — $5.99\n"
                "9. Balsamic Vinegar, 8 oz — $9.99\n"
                "10. Organic Whole Milk, 1 gal — $6.99\n"
                "11. Dijon Mustard, 8 oz — $4.49\n"
                "12. Organic Bananas, 15 lb — $14.85\n\n"
                "Subtotal: $108.72\n"
                "Delivery: Free\n"
                "Total: $108.72\n\n"
                "Everything look good?",
 'required_function_call': {'name': 'verify_details',
                            'args': {'order_id': 'ORD-5521'}},
 'function_call_response': {'status': 'success',
                            'order': {
                                'order_id': 'ORD-5521',
                                'customer': 'Carmen Reyes',
                                'phone': '818-940-3670',
                                'delivery_address': '3840 Willow Lane, Pasadena',
                                'items': [
                                    {'item_id': '1010', 'name': 'All-Purpose Flour',
                                     'qty': 2, 'price': 6.99},
                                    {'item_id': '5010', 'name': 'Sourdough Loaf',
                                     'qty': 1, 'price': 7.50},
                                    {'item_id': '2015', 'name': 'Organic Eggs',
                                     'qty': 2, 'price': 6.49},
                                    {'item_id': '3010', 'name': 'Gray Sea Salt',
                                     'qty': 1, 'price': 8.49},
                                    {'item_id': '1330', 'name': 'Maple Syrup (Grade A)',
                                     'qty': 1, 'price': 9.99},
                                    {'item_id': '4051', 'name': 'Kalamata Olives',
                                     'qty': 1, 'price': 8.49},
                                    {'item_id': '6020', 'name': 'Italian Parsley',
                                     'qty': 2, 'price': 2.49},
                                    {'item_id': '3020', 'name': 'Italian Seasoning',
                                     'qty': 1, 'price': 5.99},
                                    {'item_id': '4020', 'name': 'Balsamic Vinegar',
                                     'qty': 1, 'price': 9.99},
                                    {'item_id': '2010', 'name': 'Organic Whole Milk',
                                     'qty': 1, 'price': 6.99},
                                    {'item_id': '4025', 'name': 'Dijon Mustard',
                                     'qty': 1, 'price': 4.49},
                                    {'item_id': '6015', 'name': 'Organic Bananas',
                                     'qty': 15, 'price': 0.99}],
                                'subtotal': 108.72,
                                'delivery_fee': 0.00,
                                'total': 108.72}},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_028.wav'},

# Turn 29 — Closing: user confirms, end session
{'input': "That's perfect. Thanks so much!",
 'golden_text': "Your order ORD-5521 is confirmed! Delivery to 3840 "
                "Willow Lane, Pasadena. Your total is one hundred eight "
                "seventy-two. You'll receive a confirmation at "
                "818-940-3670. Thank you for shopping with Harvest & "
                "Hearth Market, Carmen. Have a great day!",
 'required_function_call': {'name': 'end_session', 'args': {}},
 'function_call_response': None,
 'categories': ['tool_use'],
 'audio_file': 'audio/turn_029.wav'},
]

# ============================================================================
# SCORING DIMENSIONS
#   Core (every turn):  KB · Instr · Tool
#   Conditional:        State · Ambiguity  (added per-category)
# ============================================================================

CORE_DIMENSIONS = ['kb_grounding', 'instruction_following', 'tool_use_correct']

CATEGORY_DIMENSIONS = {
    'long_range_memory':    ['state_tracking'],
    'error_recovery':       ['state_tracking'],

    'ambiguous_entity':     ['ambiguity_handling'],
    'numerical_reasoning':  ['ambiguity_handling'],

    'basic_qa':             [],
    'tool_use':             [],
}


def get_relevant_dimensions(turn: dict) -> list[str]:
    """Return scoring dimensions for a turn.

    Always includes kb_grounding, instruction_following, tool_use_correct.
    Adds state_tracking and/or ambiguity_handling based on the turn's categories.
    """
    dims = list(CORE_DIMENSIONS)

    categories = turn.get('categories', [])
    if not categories:
        cat = turn.get('category')
        categories = [cat] if cat else []

    extra = set()
    for cat in categories:
        extra.update(CATEGORY_DIMENSIONS.get(cat, []))
    dims.extend(sorted(extra))

    return dims
