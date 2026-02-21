"""
Appointment scheduling benchmark turns (25 turns: 0-24).

Scenario: A caller books TWO appointments — a dental cleaning for himself
(Daniel Nolan) and an orthodontic consultation for his daughter (Danielle
Nolan). 

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
# TURNS 0-2: SETUP — TWO PATIENTS, DOCTOR ASSIGNMENT, SCHEDULING
# ============================================================================

turns = [
# Turn 0 — Two patients in one call + negation + KB doctor specialties
{'input': "Hi, I need two appointments — a cleaning for myself and an "
          "orthodontic consultation for my daughter. But not with Dr. Perry "
          "for the braces — I've heard she's just general dentistry.",
 'golden_text': "That's right — Dr. Perry handles general dentistry including "
                "cleanings, while Dr. Barry is our orthodontist for braces "
                "and Invisalign. I can set up both for you. Which days work "
                "best?",
 'required_function_call': None,
 'categories': ['basic_qa', 'negation_reasoning'],
 'audio_file': 'audio/turn_000.wav'},

# Turn 1 — Confusable names (Daniel/Danielle) + schedule conflict + insurance
{'input': "I'm Daniel Nolan and my daughter is Danielle. We have Delta "
          "Dental. Can we do both on Thursday? She gets out of school at "
          "two-thirty.",
 'golden_text': "Hi Daniel! We do accept Delta Dental. However, Dr. Barry the "
                "orthodontist isn't in on Thursdays — he works Mondays, "
                "Wednesdays, and Fridays. Dr. Perry is available Thursday "
                "though. Would you like to find a day when both doctors are "
                "in?",
 'required_function_call': None,
 'categories': ['basic_qa', 'long_range_memory'],
 'audio_file': 'audio/turn_001.wav'},

# Turn 2 — Cross-reference two doctors' schedules from KB
{'input': "Yeah, what's the soonest day they're both in?",
 'golden_text': "The only day both Dr. Perry and Dr. Barry are in the office "
                "is Monday. The next Monday would be January 13th. Would "
                "that work?",
 'required_function_call': None,
 'categories': ['basic_qa', 'long_range_memory'],
 'audio_file': 'audio/turn_002.wav'},

# ============================================================================
# TURNS 3-6: AVAILABILITY, SLOT SELECTION, DURATION MATH
# ============================================================================

# Turn 3 — Tool call: check availability for both doctors on one date
{'input': "Monday's fine. I need morning for my cleaning — before eleven. "
          "And Danielle can't come until after three.",
 'golden_text': "Let me check Monday availability. Dr. Perry has morning "
                "slots at 9:00, 9:15, and 10:30 AM for your cleaning. For "
                "Danielle, Dr. Barry has afternoon openings at 3:15, 3:30, "
                "3:45, and 4:00 PM. Which times would you like?",
 'required_function_call': {'name': 'check_availability',
                            'args': {'date': '2025-01-13'}},
 'function_call_response': {'status': 'success',
                            'slots': [{'time': '9:00', 'doctor': 'Perry'},
                                      {'time': '9:15', 'doctor': 'Perry'},
                                      {'time': '10:30', 'doctor': 'Perry'},
                                      {'time': '15:15', 'doctor': 'Barry'},
                                      {'time': '15:30', 'doctor': 'Barry'},
                                      {'time': '15:45', 'doctor': 'Barry'},
                                      {'time': '16:00', 'doctor': 'Barry'}]},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_003.wav'},

# Turn 4 — Two slot selections in one utterance
{'input': "I'll take the nine-fifteen for me. And for Danielle, three-thirty",
 'golden_text': "Got it — 9:15 AM with Dr. Perry for your cleaning, and "
                "3:30 PM with Dr. Barry for Danielle's orthodontic "
                "consultation. Both on Monday, January 13th. I'll just "
                "need phone numbers for each to book those.",
 'required_function_call': None,
 'categories': ['numerical_reasoning', 'long_range_memory'],
 'audio_file': 'audio/turn_004.wav'},

# Turn 5 — P/B audio confusion: Perry vs Barry disambiguation
{'input': "Wait — for Danielle's consultation, that's Dr. Barry, B as in "
          "boy? Not Perry? I keep mixing them up.",
 'golden_text': "Correct — Dr. Barry, B as in boy, is the orthodontist "
                "doing Danielle's consultation. Dr. Perry, P as in Paul, "
                "will do your cleaning. Different doctors for each "
                "appointment.",
 'required_function_call': None,
 'categories': ['ambiguous_entity'],
 'audio_file': 'audio/turn_005.wav'},

# Turn 6 — Duration math: 3:30 + 45 min vs 5:00 deadline
{'input': "How long is the ortho consultation? Because Danielle has piano "
          "at five-fifteen and we need to leave by five.",
 'golden_text': "An orthodontic consultation is 45 minutes. Starting at 3:30, "
                "she'd be done by 4:15 PM — plenty of time to leave by 5:00. "
                "Our last appointment slot is 4:15 anyway.",
 'required_function_call': None,
 'categories': ['numerical_reasoning'],
 'audio_file': 'audio/turn_006.wav'},

# ============================================================================
# TURNS 7-9: CONTACT DETAILS, NAME SPELLING, KB QUESTION
# ============================================================================

# Turn 7 — Two similar phone numbers (916/960, 1640/1614)
{'input': "OK great. My cell is 415-916-1640. For Danielle, use my wife's "
          "number — 415-960-1614. Don't book yet though — I have a few "
          "more questions first.",
 'golden_text': "Let me confirm: your number, Daniel, is 415-916-1640. And "
                "for Danielle, your wife's number: 415-960-1614. Is that "
                "right?",
 'required_function_call': None,
 'categories': ['numerical_reasoning', 'long_range_memory'],
 'audio_file': 'audio/turn_007.wav'},

# Turn 8 — Last name spelling confirmation (M/N audio trap)
{'input': "Last name for both is Nolan — N as in November, O-L-A-N.",
 'golden_text': "Yes, I have Nolan, N-O-L-A-N, for both Daniel and Danielle. "
                "Thanks for confirming the spelling.",
 'required_function_call': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_008.wav'},

# Turn 9 — Off-topic KB question mid-booking (parking info)
{'input': "Before we book — where do I park? And is there a fee?",
 'golden_text': "We have a free parking lot behind the building — about 30 "
                "spaces, entrance on Oak Avenue. Street parking on Elm "
                "Street is metered with a 2-hour limit. No validation "
                "needed since our lot is free.",
 'required_function_call': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_009.wav'},

# ============================================================================
# TURNS 10-12: BOOKING + SLOT-TAKEN ERROR RECOVERY
# ============================================================================

# Turn 10 — Book Daniel's cleaning (first tool-call booking)
{'input': "Perfect. Let's book mine first — the Monday morning cleaning.",
 'golden_text': "Booked! Your cleaning with Dr. Perry is confirmed for "
                "Monday, January 13th at 9:15 AM. Appointment ID: APT-001.",
 'required_function_call': {'name': 'book_appointment',
                            'args': {'patient_name': 'Daniel Nolan',
                                     'phone': '415-916-1640',
                                     'date': '2025-01-13',
                                     'time': '09:15',
                                     'doctor': 'Perry',
                                     'service_type': 'cleaning'}},
 'function_call_response': {'status': 'success',
                            'appointment_id': 'APT-001'},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_010.wav'},

# Turn 11 — Book Danielle's consultation → SLOT_TAKEN error recovery
{'input': "Now book Danielle's ortho consultation.",
 'golden_text': "I'm sorry — the 3:30 PM slot with Dr. Barry on Monday was "
                "just taken by another patient. The remaining Monday "
                "afternoon slots with Dr. Barry are 3:15, 3:45, and "
                "4:00 PM. Which would you prefer?",
 'required_function_call': {'name': 'book_appointment',
                            'args': {'patient_name': 'Danielle Nolan',
                                     'phone': '415-960-1614',
                                     'date': '2025-01-13',
                                     'time': '15:30',
                                     'doctor': 'Barry',
                                     'service_type': 'orthodontic consultation'}},
 'function_call_response': {'status': 'error',
                            'error_code': 'SLOT_TAKEN',
                            'message': 'The 15:30 slot on 2025-01-13 is no longer available'},
 'categories': ['tool_use', 'error_recovery', 'long_range_memory'],
 'audio_file': 'audio/turn_011.wav'},

# Turn 12 — Rebook at new time + verify duration fits before 5:00
{'input': "Three-forty-five. She'll still be done before five "
          "with a forty-five minute consultation, right?",
 'golden_text': "Yes — 3:45 plus 45 minutes puts her at 4:30 PM, well before "
                "5:00. Booked! Danielle's orthodontic consultation with "
                "Dr. Barry: Monday, January 13th at 3:45 PM. "
                "Appointment ID: APT-002.",
 'required_function_call': {'name': 'book_appointment',
                            'args': {'patient_name': 'Danielle Nolan',
                                     'phone': '415-960-1614',
                                     'date': '2025-01-13',
                                     'time': '15:45',
                                     'doctor': 'Barry',
                                     'service_type': 'orthodontic consultation'}},
 'function_call_response': {'status': 'success',
                            'appointment_id': 'APT-002'},
 'categories': ['tool_use', 'numerical_reasoning', 'long_range_memory'],
 'audio_file': 'audio/turn_012.wav'},

# ============================================================================
# TURNS 13-15: TIME CHANGE, FALSE MEMORY TRAP, HESITATION
# ============================================================================

# Turn 13 — Move Daniel's time; Danielle's must stay unchanged
{'input': "Actually, I just remembered I have a nine AM call. Move my "
          "cleaning to ten-thirty.",
 'golden_text': "Updated — your cleaning is now at 10:30 AM instead of "
                "9:15 AM on Monday with Dr. Perry. Danielle's 3:45 PM "
                "consultation with Dr. Barry stays the same.",
 'required_function_call': {'name': 'update_patient_info',
                            'args': {'appointment_id': 'APT-001',
                                     'field': 'time',
                                     'new_value': '10:30'}},
 'function_call_response': {'status': 'success'},
 'categories': ['tool_use', 'long_range_memory', 'implicit_correction'],
 'audio_file': 'audio/turn_013.wav'},

# Turn 14 — False memory trap: user claims a slot that was never offered
{'input': "Hmm, I'm second-guessing the ten-thirty. What were the other "
          "morning slots you mentioned? I remember nine and eight-forty-five.",
 'golden_text': "The other morning slots were 9:00 AM and 9:15 AM. There "
                "was no 8:45 slot — those were the only three options "
                "for Dr. Perry on Monday morning.",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_014.wav'},

# Turn 15 — Hesitation then confirmation: keep current time, no change
{'input': "Oh right, nine and nine-fifteen. Nah, keep it at ten-thirty. "
          "That's fine.",
 'golden_text': "OK, keeping your cleaning at 10:30 AM with Dr. Perry.",
 'required_function_call': None,
 'categories': ['basic_qa', 'long_range_memory'],
 'audio_file': 'audio/turn_015.wav'},

# ============================================================================
# TURNS 16-17: PHONE NUMBER SWAP + SHORT CONFIRMATION
# ============================================================================

# Turn 16 — Swap phones across two appointments (two tool calls)
{'input': "Actually, I want to swap the phone numbers. Put my wife's cell "
          "on my appointment and my number on Danielle's.",
 'golden_text': "Done — swapped both phone numbers. Your cleaning appointment "
                "now has 415-960-1614, and Danielle's consultation now has "
                "415-916-1640.",
 'required_function_call': [
     {'name': 'update_patient_info',
      'args': {'appointment_id': 'APT-001',
               'field': 'phone',
               'new_value': '415-960-1614'}},
     {'name': 'update_patient_info',
      'args': {'appointment_id': 'APT-002',
               'field': 'phone',
               'new_value': '415-916-1640'}},
 ],
 'function_call_response': {'status': 'success'},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_016.wav'},

# Turn 17 — Minimal one-word confirmation (short audio)
{'input': "Yep.",
 'golden_text': "Great. Is there anything else you'd like to change, or are "
                "we good to go?",
 'required_function_call': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_017.wav'},

# ============================================================================
# TURNS 18-20: THREE CONSECUTIVE FALSE MEMORY TRAPS
# ============================================================================

# Turn 18 — False memory trap: user states wrong day (Thursday vs Monday)
{'input': "So just to confirm — my cleaning is Thursday at ten-thirty and "
          "Danielle's consultation is Monday at three-forty-five?",
 'golden_text': "Almost — both appointments are on Monday, January 13th. "
                "Your cleaning is at 10:30 AM with Dr. Perry, and "
                "Danielle's consultation is at 3:45 PM with Dr. Barry. "
                "There is no Thursday appointment.",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_018.wav'},

# Turn 19 — False memory trap: user states wrong doctor for Danielle
{'input': "Right, both Monday. And Danielle's seeing Dr. Perry for the "
          "consultation, correct?",
 'golden_text': "No — Danielle is seeing Dr. Barry, B as in boy, for the "
                "orthodontic consultation. Dr. Perry is doing your cleaning.",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_019.wav'},

# Turn 20 — False memory trap: user states wrong phone number post-swap
{'input': "Got it. And the phone numbers — I have 415-916-1640 on my "
          "appointment, right?",
 'golden_text': "Actually no — after the swap, your appointment has "
                "415-960-1614 and Danielle's has 415-916-1640. Would you "
                "like to change anything?",
 'required_function_call': None,
 'categories': ['long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_020.wav'},

# ============================================================================
# TURN 21: REVERT PHONE SWAP
# ============================================================================

# Turn 21 — Undo phone swap: restore original numbers (two tool calls)
{'input': "Ugh, that's confusing. Put them back the way they were — my cell "
          "on mine, wife's number on Danielle's.",
 'golden_text': "Done — reverted both phone numbers. Your cleaning appointment "
                "is back to 415-916-1640, and Danielle's consultation is back "
                "to 415-960-1614. Both restored to the originals.",
 'required_function_call': [
     {'name': 'update_patient_info',
      'args': {'appointment_id': 'APT-001',
               'field': 'phone',
               'new_value': '415-916-1640'}},
     {'name': 'update_patient_info',
      'args': {'appointment_id': 'APT-002',
               'field': 'phone',
               'new_value': '415-960-1614'}},
 ],
 'function_call_response': {'status': 'success'},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_021.wav'},

# ============================================================================
# TURNS 22-24: RECAP, KB QUESTION, CLOSING
# ============================================================================

# Turn 22 — Full recap of both appointments after all revisions
{'input': "Read me back everything — both appointments, all the details.",
 'golden_text': "Here's everything:\n\n"
                "1. Daniel Nolan — Cleaning with Dr. Perry, Monday "
                "January 13th at 10:30 AM. Phone: 415-916-1640.\n\n"
                "2. Danielle Nolan — Orthodontic consultation with "
                "Dr. Barry, Monday January 13th at 3:45 PM. Phone: "
                "415-960-1614.\n\n"
                "Both under Delta Dental. Is everything correct?",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_022.wav'},

# Turn 23 — KB question: first-time patient instructions for both
{'input': "Perfect. We haven't been before — anything we should know?",
 'golden_text': "Since you're both first-time patients, please arrive 15 "
                "minutes early for intake paperwork — that's 10:15 AM for "
                "you and 3:30 PM for Danielle. You can download the forms "
                "ahead of time at bayshorefamilydental.com. Anything else "
                "I can help with?",
 'required_function_call': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_023.wav'},

# Turn 24 — Closing: user confirms done, end session
{'input': "Great, that's everything. Thanks!",
 'golden_text': "You're welcome! See you both on Monday. Have a great day!",
 'required_function_call': {'name': 'end_session', 'args': {}},
 'categories': ['tool_use'],
 'audio_file': 'audio/turn_024.wav'},
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
    'negation_reasoning':   [],
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
