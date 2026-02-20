"""
Event planning benchmark turns (29 turns: 0-28).

Scenario: Priya Mehta is planning her company's annual gala. She starts with
80 guests at the Garden Pavilion, upgrades to 120 (requiring the Grand
Ballroom), then downsizes back to 90 (returning to the Garden Pavilion).
Along the way she selects catering, adds services, adjusts her budget,
encounters three false memory traps, changes the contact number, and
reschedules the date.

Scoring dimensions:
  Core (every turn):
  - KB:        KB / factual grounding
  - Instr:     Instruction following
  - Tool:      Tool-use correctness
  Conditional (added per-category):
  - State:     State tracking across turns
  - Ambiguity: Ambiguity handling (entities, numbers, context)

Usage:
    from turns import turns, get_relevant_dimensions
"""

# ============================================================================
# TURNS 0-6: SETUP — INQUIRY, VENUE SEARCH, CATERING, INITIAL QUOTE
# ============================================================================

turns = [
# Turn 0 — Opening: guest count
{'input': "Hi, I'm planning our annual company gala for about eighty people. "
          "I'd like to know what venue options you have available.",
 'golden_text': "Welcome! I'd love to help you plan your company gala for "
                "eighty guests. Do you have a date in mind? That will help "
                "me check venue availability for you.",
 'required_function_call': None,
 'categories': ['basic_qa', 'numerical_reasoning'],
 'audio_file': 'audio/turn_000.wav'},

# Turn 1 — Date preference: Saturday March 8th
{'input': "We're looking at Saturday, March 8th. What venues do you have "
          "that could fit eighty people?",
 'golden_text': "Let me search our venues for March 8th with eighty guests.",
 'required_function_call': {'name': 'search_venues',
                            'args': {'date': '2025-03-08',
                                     'guest_count': 80}},
 'function_call_response': {'status': 'success',
                            'venues': [
                                {'venue_id': 'rooftop_terrace',
                                 'name': 'The Rooftop Terrace',
                                 'capacity': 50,
                                 'rental_fee': 2500,
                                 'available': True,
                                 'fits_guest_count': False},
                                {'venue_id': 'garden_pavilion',
                                 'name': 'The Garden Pavilion',
                                 'capacity': 100,
                                 'rental_fee': 4000,
                                 'available': True,
                                 'fits_guest_count': True},
                                {'venue_id': 'grand_ballroom',
                                 'name': 'The Grand Ballroom',
                                 'capacity': 250,
                                 'rental_fee': 8000,
                                 'available': True,
                                 'fits_guest_count': True}]},
 'categories': ['tool_use'],
 'audio_file': 'audio/turn_001.wav'},

# Turn 2 — Present venue results and recommend Garden Pavilion
{'input': "Tell me about those options.",
 'golden_text': "All three venues are available on March 8th, but the "
                "Rooftop Terrace only holds 50 so it's too small for eighty "
                "guests. Your two options are:\n\n"
                "1. The Garden Pavilion — up to 100 guests, $4,000 rental. "
                "Beautiful outdoor pavilion with a covered area and dance "
                "floor.\n\n"
                "2. The Grand Ballroom — up to 250 guests, $8,000 rental. "
                "Elegant indoor ballroom with a full AV system.\n\n"
                "The Garden Pavilion is a great fit for eighty guests and "
                "would save you $4,000 over the ballroom. Would you like "
                "more details on either?",
 'required_function_call': None,
 'categories': ['basic_qa', 'numerical_reasoning'],
 'audio_file': 'audio/turn_002.wav'},

# Turn 3 — ENHANCED #2: Vague pronoun — "the second one, the outdoor one"
{'input': "The second one — the outdoor one for four thousand — what "
          "exactly does it include? Capacity, amenities, that sort of thing.",
 'golden_text': "The Garden Pavilion holds up to 100 guests and rents for "
                "$4,000. It's a beautifully landscaped outdoor pavilion "
                "surrounded by native Texas gardens. Amenities include a "
                "covered pavilion with ceiling fans, a dance floor, a "
                "built-in stage area, a full lighting package, on-site "
                "restrooms, and wheelchair accessibility. There's also a "
                "backup indoor option at no extra charge in case of rain. "
                "Setup and teardown are included.",
 'required_function_call': None,
 'categories': ['basic_qa', 'ambiguous_entity'],
 'audio_file': 'audio/turn_003.wav'},

# Turn 4 — Provide name and phone number
{'input': "That sounds perfect. I'm Priya Mehta — M-E-H-T-A. My number is "
          "512-847-3160.",
 'golden_text': "Got it — Priya Mehta, M-E-H-T-A. And your phone number is "
                "five-one-two, eight-four-seven, three-one-six-zero. Is that "
                "correct?",
 'required_function_call': None,
 'categories': ['basic_qa', 'numerical_reasoning'],
 'audio_file': 'audio/turn_004.wav'},

# Turn 5 — Ask about catering packages
{'input': "That's right. Now what catering packages do you offer? I heard "
          "you have one around forty-five dollars per person.",
 'golden_text': "We have three catering packages. The Silver is $45 per "
                "person and includes a choice "
                "of 3 passed appetizers, 2 entrées, and non-alcoholic "
                "beverages. The Gold is $75 per person with 4 appetizers, "
                "a cheese and charcuterie display, 3 entrées, a dessert "
                "bar, and a wine and beer bar. The Platinum is $110 per "
                "person with a full plated course, open bar, custom dessert "
                "station, and a dedicated event coordinator included. Would "
                "you like more details on any of them?",
 'required_function_call': None,
 'categories': ['basic_qa', 'numerical_reasoning'],
 'audio_file': 'audio/turn_005.wav'},

# Turn 6 — ENHANCED #1: Mid-sentence self-correction (Platinum → Gold)
{'input': "Let's go with the Platinum— actually no, the Gold package. At "
          "seventy-five per person for eighty guests. Can you give me a "
          "quote for that with the Garden Pavilion?",
 'golden_text': "Got it — the Gold package, not Platinum. Let me get a "
                "quote for the Garden Pavilion with eighty guests and Gold "
                "catering.",
 'required_function_call': {'name': 'get_quote',
                            'args': {'venue_id': 'garden_pavilion',
                                     'guest_count': 80,
                                     'catering_package': 'gold',
                                     'add_ons': []}},
 'function_call_response': {'status': 'success',
                            'quote': {
                                'venue_rental': 4000,
                                'catering': 6000,
                                'catering_per_person': 75,
                                'add_ons_total': 0,
                                'add_ons_breakdown': {},
                                'total': 10000}},
 'categories': ['tool_use', 'numerical_reasoning', 'implicit_correction'],
 'audio_file': 'audio/turn_006.wav'},

# ============================================================================
# TURNS 7-15: MIDGAME — BUDGET, ADD-ONS, BOOKING, CHANGES
# ============================================================================

# Turn 7 — ENHANCED #3: User provides wrong math, model must correct
{'input': "So let me make sure I have this right — the venue is four "
          "thousand, and catering at seventy-five a head for eighty people "
          "is... five thousand five hundred? So about nine-five total?",
 'golden_text': "Actually, the catering comes to $6,000 — that's $75 times "
                "80 guests. With the $4,000 venue rental, the total is "
                "$10,000, not $9,500. No add-ons are included yet. Would "
                "you like to add any services like a DJ, photographer, or "
                "floral arrangements?",
 'required_function_call': None,
 'categories': ['numerical_reasoning', 'error_recovery'],
 'audio_file': 'audio/turn_007.wav'},

# Turn 8 — ENHANCED #8: Ambiguous add-on — "music option" (DJ vs band)
{'input': "Yeah, we'll need the music option for sure. And a photographer.",
 'golden_text': "We have two music options — a professional DJ for $800 "
                "which includes a full sound system, lighting effects, and "
                "MC services for up to 5 hours, or a live 5-piece band for "
                "$2,500 doing a 3-hour set. Which would you prefer? I can "
                "also add the photographer at $1,200 for up to 6 hours "
                "with 300-plus edited digital photos.",
 'required_function_call': None,
 'categories': ['basic_qa', 'ambiguous_entity'],
 'audio_file': 'audio/turn_008.wav'},

# Turn 9 — ENHANCED #4: Multi-request with mid-stream reversal
{'input': "The DJ, not the band. And yes the photographer. Oh and throw "
          "in floral arrangements too — actually wait, no, skip the flowers. "
          "Budget's tight. Just the DJ and photographer. Can I get an "
          "updated quote?",
 'golden_text': "Got it — DJ and photographer only, no floral arrangements. "
                "Let me get an updated quote.",
 'required_function_call': {'name': 'get_quote',
                            'args': {'venue_id': 'garden_pavilion',
                                     'guest_count': 80,
                                     'catering_package': 'gold',
                                     'add_ons': ['dj', 'photographer']}},
 'function_call_response': {'status': 'success',
                            'quote': {
                                'venue_rental': 4000,
                                'catering': 6000,
                                'catering_per_person': 75,
                                'add_ons_total': 2000,
                                'add_ons_breakdown': {'dj': 800,
                                                      'photographer': 1200},
                                'total': 12000}},
 'categories': ['tool_use', 'numerical_reasoning', 'implicit_correction'],
 'audio_file': 'audio/turn_009.wav'},

# Turn 10 — Book the event
{'input': "Twelve thousand works for our budget. Let's book it.",
 'golden_text': "Your event is booked! Here are the details: Priya Mehta, "
                "Garden Pavilion, Saturday March 8th, 80 guests, Gold "
                "catering, DJ and photographer. Total: $12,000. Your "
                "event ID is EVT-3001.",
 'required_function_call': {'name': 'book_event',
                            'args': {'name': 'Priya Mehta',
                                     'phone': '512-847-3160',
                                     'date': '2025-03-08',
                                     'venue_id': 'garden_pavilion',
                                     'guest_count': 80,
                                     'catering_package': 'gold',
                                     'add_ons': ['dj', 'photographer']}},
 'function_call_response': {'status': 'success',
                            'event_id': 'EVT-3001',
                            'total': 12000},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_010.wav'},

# Turn 11 — FALSE MEMORY TRAP #1: wrong venue
{'input': "Great. So we booked the Rooftop Terrace, right?",
 'golden_text': "Actually, no — you booked the Garden Pavilion, not the "
                "Rooftop Terrace. The Rooftop Terrace only holds 50 guests, "
                "which wouldn't fit your party of 80. Your event is confirmed "
                "at the Garden Pavilion with a capacity of 100.",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_011.wav'},

# Turn 12 — ENHANCED #9: Nested self-correction (150 → 120) + capacity check
{'input': "Right, the Garden Pavilion. Actually, I just got word — we're "
          "up to a hundred and fif— sorry, a hundred and twenty. One-twenty. "
          "Does the Pavilion still work for that?",
 'golden_text': "The Garden Pavilion has a maximum capacity of 100 guests, "
                "so 120 won't fit there unfortunately. We'd need to move to "
                "the Grand Ballroom, which holds up to 250 guests. The "
                "rental is $8,000 instead of $4,000. Would you like me to "
                "look into switching?",
 'required_function_call': None,
 'categories': ['numerical_reasoning', 'long_range_memory', 'error_recovery',
                'implicit_correction'],
 'audio_file': 'audio/turn_012.wav'},

# Turn 13 — Confirm switch to Grand Ballroom
{'input': "Yeah, we'll have to go with the Grand Ballroom then. Can you "
          "switch us over?",
 'golden_text': "Updated — your event is now at the Grand Ballroom instead "
                "of the Garden Pavilion, with 120 guests.",
 'required_function_call': [
     {'name': 'update_event',
      'args': {'event_id': 'EVT-3001',
               'field': 'venue_id',
               'new_value': 'grand_ballroom'}},
     {'name': 'update_event',
      'args': {'event_id': 'EVT-3001',
               'field': 'guest_count',
               'new_value': '120'}}],
 'function_call_response': [
     {'status': 'success'},
     {'status': 'success'}],
 'categories': ['tool_use', 'long_range_memory', 'implicit_correction'],
 'audio_file': 'audio/turn_013.wav'},

# Turn 14 — Get updated quote for Grand Ballroom with 120 guests
{'input': "What does the total look like now with the bigger venue and "
          "more guests?",
 'golden_text': "Let me get you an updated quote.",
 'required_function_call': {'name': 'get_quote',
                            'args': {'venue_id': 'grand_ballroom',
                                     'guest_count': 120,
                                     'catering_package': 'gold',
                                     'add_ons': ['dj', 'photographer']}},
 'function_call_response': {'status': 'success',
                            'quote': {
                                'venue_rental': 8000,
                                'catering': 9000,
                                'catering_per_person': 75,
                                'add_ons_total': 2000,
                                'add_ons_breakdown': {'dj': 800,
                                                      'photographer': 1200},
                                'total': 19000}},
 'categories': ['tool_use', 'numerical_reasoning'],
 'audio_file': 'audio/turn_014.wav'},

# Turn 15 — FALSE MEMORY TRAP #2: wrong catering package
{'input': "Nineteen thousand. The catering is the Silver package, right?",
 'golden_text': "No — you selected the Gold package, not Silver. Gold is "
                "$75 per person and includes appetizers, a cheese and "
                "charcuterie display, three entrées, a dessert bar, and a "
                "wine and beer bar. Silver is $45 per person with fewer "
                "options. Would you like to keep Gold or switch to Silver "
                "to reduce costs?",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_015.wav'},

# ============================================================================
# TURNS 16-28: ENDGAME — BUDGET, HYPOTHETICAL, DOWNSIZE, UPDATES, CLOSING
# ============================================================================

# Turn 16 — Budget concern, request breakdown
{'input': "No, keep Gold. But nineteen thousand is way over our original "
          "budget of twelve. Can you show me the breakdown?",
 'golden_text': "Here's the breakdown for 120 guests at the Grand Ballroom "
                "with Gold catering: venue rental is $8,000 (up from $4,000 "
                "at the Garden Pavilion), Gold catering at $75 per person "
                "for 120 guests is $9,000 (up from $6,000 for 80), and "
                "add-ons are $2,000 — $800 for the DJ and $1,200 for the "
                "photographer. That totals $19,000, which is $7,000 more "
                "than your original $12,000 booking.",
 'required_function_call': None,
 'categories': ['numerical_reasoning', 'long_range_memory'],
 'audio_file': 'audio/turn_016.wav'},

# Turn 17 — ENHANCED #7: Hypothetical reasoning (no actual changes)
{'input': "Hmm. Before I decide anything — what if we just dropped the "
          "photographer and switched to Silver? Would that get us back "
          "under twelve thousand?",
 'golden_text': "Let me work that out. With Silver catering at $45 per "
                "person for 120 guests, that's $5,400. The Grand Ballroom "
                "is $8,000, and just the DJ at $800 — that totals $14,200. "
                "So even with those changes, you'd still be $2,200 over "
                "your $12,000 target. The biggest cost driver is the venue "
                "upgrade from $4,000 to $8,000.",
 'required_function_call': None,
 'categories': ['numerical_reasoning', 'long_range_memory'],
 'audio_file': 'audio/turn_017.wav'},

# Turn 18 — Downsize to 90 guests
{'input': "Yeah that's still too much. What if we cut back to ninety "
          "people instead? Would that fit at the Garden Pavilion again?",
 'golden_text': "Yes — 90 guests fits within the Garden Pavilion's capacity "
                "of 100. That would bring your venue cost back down to "
                "$4,000 instead of $8,000. And Gold catering at $75 per "
                "person for 90 guests would be $6,750. Shall I switch you "
                "back to the Garden Pavilion with 90 guests?",
 'required_function_call': None,
 'categories': ['numerical_reasoning', 'long_range_memory'],
 'audio_file': 'audio/turn_018.wav'},

# Turn 19 — Switch back to Garden Pavilion with 90 guests
{'input': "Yes, switch us back to the Garden Pavilion with ninety guests.",
 'golden_text': "Done — your event is back at the Garden Pavilion with 90 "
                "guests.",
 'required_function_call': [
     {'name': 'update_event',
      'args': {'event_id': 'EVT-3001',
               'field': 'venue_id',
               'new_value': 'garden_pavilion'}},
     {'name': 'update_event',
      'args': {'event_id': 'EVT-3001',
               'field': 'guest_count',
               'new_value': '90'}}],
 'function_call_response': [
     {'status': 'success'},
     {'status': 'success'}],
 'categories': ['tool_use', 'long_range_memory', 'implicit_correction'],
 'audio_file': 'audio/turn_019.wav'},

# Turn 20 — Updated quote for 90 guests at Garden Pavilion
{'input': "What's the new total?",
 'golden_text': "Let me get you the updated quote.",
 'required_function_call': {'name': 'get_quote',
                            'args': {'venue_id': 'garden_pavilion',
                                     'guest_count': 90,
                                     'catering_package': 'gold',
                                     'add_ons': ['dj', 'photographer']}},
 'function_call_response': {'status': 'success',
                            'quote': {
                                'venue_rental': 4000,
                                'catering': 6750,
                                'catering_per_person': 75,
                                'add_ons_total': 2000,
                                'add_ons_breakdown': {'dj': 800,
                                                      'photographer': 1200},
                                'total': 12750}},
 'categories': ['tool_use', 'numerical_reasoning'],
 'audio_file': 'audio/turn_020.wav'},

# Turn 21 — ENHANCED #10: Contradictory callback with dual error
{'input': "OK much better. So the DJ is eight hundred, and the "
          "photographer is... fifteen hundred? That's twenty-three hundred "
          "in add-ons?",
 'golden_text': "Not quite — the photographer is $1,200, not $1,500. So "
                "your add-ons total $2,000: $800 for the DJ plus $1,200 "
                "for the photographer. That's $2,000, not $2,300.",
 'required_function_call': None,
 'categories': ['long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_021.wav'},

# Turn 22 — ENHANCED #5: Phone number swap to assistant
{'input': "Got it, thanks. Oh — one more thing. Can you change the contact "
          "number on the booking? Use my assistant Raj's number instead: "
          "512-849-3016.",
 'golden_text': "Updated — the contact number on the booking is now "
                "five-one-two, eight-four-nine, three-zero-one-six for "
                "your assistant Raj.",
 'required_function_call': {'name': 'update_event',
                            'args': {'event_id': 'EVT-3001',
                                     'field': 'phone',
                                     'new_value': '512-849-3016'}},
 'function_call_response': {'status': 'success'},
 'categories': ['tool_use', 'long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_022.wav'},

# Turn 23 — ENHANCED #6: Retroactive date change (search new date)
{'input': "And I just realized — March 8th actually conflicts with our "
          "company retreat. Can we move the whole thing to March 15th? "
          "Is the Garden Pavilion available that Saturday?",
 'golden_text': "Let me check availability for March 15th.",
 'required_function_call': {'name': 'search_venues',
                            'args': {'date': '2025-03-15',
                                     'guest_count': 90}},
 'function_call_response': {'status': 'success',
                            'venues': [
                                {'venue_id': 'rooftop_terrace',
                                 'name': 'The Rooftop Terrace',
                                 'capacity': 50,
                                 'rental_fee': 2500,
                                 'available': True,
                                 'fits_guest_count': False},
                                {'venue_id': 'garden_pavilion',
                                 'name': 'The Garden Pavilion',
                                 'capacity': 100,
                                 'rental_fee': 4000,
                                 'available': True,
                                 'fits_guest_count': True},
                                {'venue_id': 'grand_ballroom',
                                 'name': 'The Grand Ballroom',
                                 'capacity': 250,
                                 'rental_fee': 8000,
                                 'available': True,
                                 'fits_guest_count': True}]},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_023.wav'},

# Turn 24 — Confirm date change
{'input': "Great, it's available. Move us to March 15th.",
 'golden_text': "Done — your event date is now Saturday, March 15th at "
                "the Garden Pavilion. Everything else stays the same.",
 'required_function_call': {'name': 'update_event',
                            'args': {'event_id': 'EVT-3001',
                                     'field': 'date',
                                     'new_value': '2025-03-15'}},
 'function_call_response': {'status': 'success'},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_024.wav'},

# Turn 25 — Full recap (must reflect ALL accumulated changes)
{'input': "OK, can you read me back the full booking now? Everything "
          "from the top.",
 'golden_text': "Here's your complete booking:\n\n"
                "Event ID: EVT-3001\n"
                "Contact: Priya Mehta, c/o Raj at 512-849-3016\n"
                "Date: Saturday, March 15th, 2025\n"
                "Venue: The Garden Pavilion (capacity 100)\n"
                "Guest count: 90\n"
                "Catering: Gold package at $75 per person — $6,750\n"
                "Add-ons: DJ ($800) + Photographer ($1,200) — $2,000\n"
                "Venue rental: $4,000\n"
                "Total: $12,750\n\n"
                "Is everything correct?",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_025.wav'},

# Turn 26 — Deposit question from KB
{'input': "That's all correct. What's the deposit situation? How much do "
          "I need to pay now?",
 'golden_text': "A 50% deposit is required to secure your date. That would "
                "be $6,375 based on your $12,750 total. The remaining "
                "balance of $6,375 is due 7 days before the event — so by "
                "March 8th. We accept cash, check, all major credit cards, "
                "and bank wire transfer.",
 'required_function_call': None,
 'categories': ['basic_qa', 'numerical_reasoning'],
 'audio_file': 'audio/turn_026.wav'},

# Turn 27 — Cancellation policy from KB
{'input': "And what if we need to cancel? What's your cancellation policy?",
 'golden_text': "Our cancellation policy is based on how far in advance "
                "you cancel: 30 or more days before the event, you get a "
                "full refund of the deposit. Between 15 and 29 days, you "
                "get a 50% refund. Fewer than 15 days out, there's no "
                "refund. You also get one complimentary date change if "
                "requested 30 or more days in advance, subject to "
                "availability.",
 'required_function_call': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_027.wav'},

# Turn 28 — Final goodbye + end session
{'input': "Perfect, that all sounds good. I'll get the deposit sorted out. "
          "Thanks so much for your help!",
 'golden_text': "You're welcome, Priya! I'll send a confirmation to Raj "
                "at 512-849-3016. Your event at the Garden Pavilion on "
                "March 15th is all set. Have a great day!",
 'required_function_call': {'name': 'end_session', 'args': {}},
 'categories': ['tool_use'],
 'audio_file': 'audio/turn_028.wav'},
]

# ============================================================================
# SCORING DIMENSIONS
#   Core (every turn):  KB · Instr · Tool
#   Conditional:        State · Ambiguity  (added per-category)
# ============================================================================

CORE_DIMENSIONS = ['kb_grounding', 'instruction_following', 'tool_use_correct']

CATEGORY_DIMENSIONS = {
    'long_range_memory':    ['state_tracking'],
    'implicit_correction':  ['state_tracking'],
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
