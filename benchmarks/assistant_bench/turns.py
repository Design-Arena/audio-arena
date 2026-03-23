"""
Personal assistant conversation benchmark turns (31 turns: 0-30).

Scenario: Dana Fischer calls Atlas Personal Assistant to handle multiple
tasks in a single session. She needs to book a flight to NYC for a work
meeting, send emails to colleagues, book a hotel, set reminders, and add
calendar events. 

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
# TURNS 0-6: EARLY BUCKET — FLIGHT BOOKING + FIRST MULTI-INTENT
# ============================================================================

turns = [
# Turn 0 — MULTI-INTENT: flight + email in one turn
{'input': "Hi, this is Dana Fischer — that's F-I-S-C-H-E-R. I need to "
          "book a flight to New York and send an email to my colleague Alex "
          "about the meeting agenda.",
 'golden_text': "Hi Dana! I can help with both — the flight to New York "
                "and the email to Alex about the meeting agenda. Let's "
                "start with the flight. When are you looking to fly, and "
                "where will you be departing from?",
 'required_function_call': None,
 'categories': ['basic_qa', 'long_range_memory'],
 'audio_file': 'audio/turn_000.wav'},

# Turn 1 — Flight details + calendar check → report conflicts
{'input': "Flying out of LAX, February third — I'd like a morning flight.",
 'golden_text': "LAX to New York on February 3rd, morning flight. I checked "
                "your calendar and you have a budget review meeting from "
                "8 to 9 AM and team standup from 9 to 10 AM. A morning "
                "flight would conflict with both of those. Would you like "
                "to look at afternoon options instead?",
 'required_function_call': {'name': 'check_calendar',
                            'args': {'date': '2025-02-03'}},
 'function_call_response': {'status': 'success',
                            'events': [
                                {'time': '08:00-09:00',
                                 'title': 'Budget review meeting',
                                 'notes': 'With finance team'},
                                {'time': '09:00-10:00',
                                 'title': 'Team standup',
                                 'notes': 'Weekly recurring'}]},
 'categories': ['tool_use'],
 'audio_file': 'audio/turn_001.wav'},

# Turn 2 — User acknowledges conflicts, asks for afternoon options
{'input': "Oh right, I totally forgot about those. Yeah, what afternoon "
          "flights do you have?",
 'golden_text': "For afternoon flights from LAX to JFK, I have two options: "
                "Delta DL-582 at 2:30 PM arriving at 11 PM for $315 per "
                "person, or American AA-720 at 6:15 PM arriving at 2:45 AM "
                "for $265. Which sounds better?",
 'required_function_call': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_002.wav'},

# Turn 3 — AMBIGUOUS ENTITY: "the afternoon one" when two afternoon flights exist
{'input': "How much is the afternoon one?",
 'golden_text': "There are two afternoon options — Delta DL-582 at 2:30 PM "
                "for $315, or American AA-720 at 6:15 PM for $265. "
                "Which one are you interested in?",
 'required_function_call': None,
 'categories': ['basic_qa', 'ambiguous_entity'],
 'audio_file': 'audio/turn_003.wav'},

# Turn 4 — Book the flight (resolves Turn 3 ambiguity)
{'input': "The 2:30 Delta. Book that for just me.",
 'golden_text': "Booked! Your flight is confirmed — Dana Fischer on Delta "
                "DL-582, LAX to JFK, February 3rd at 2:30 PM. One "
                "passenger, $315. Your confirmation ID is FL-3021.",
 'required_function_call': {'name': 'book_flight',
                            'args': {'origin': 'LAX',
                                     'destination': 'JFK',
                                     'date': '2025-02-03',
                                     'time': '14:30',
                                     'passengers': 1,
                                     'passenger_name': 'Dana Fischer'}},
 'function_call_response': {'status': 'success',
                            'confirmation_id': 'FL-3021',
                            'flight_id': 'DL-582',
                            'airline': 'Delta',
                            'price': 315},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_004.wav'},

# Turn 5 — OVER-CLARIFICATION TRAP: "that email" — only one email mentioned so far
{'input': "Great. Now about that email I mentioned.",
 'golden_text': "Right — the email to Alex Reed about the meeting "
                "agenda. What would you like the email to say?",
 'required_function_call': None,
 'categories': ['long_range_memory', 'ambiguous_entity'],
 'audio_file': 'audio/turn_005.wav'},

# Turn 6 — Compose and send the email
{'input': "Just say: 'Hi Alex, I'll be in New York on February 3rd for "
          "our meeting. Can you put together the agenda and share it by "
          "end of week? Main topics should be Q4 results and the new "
          "product roadmap. Thanks, Dana.' Send that.",
 'golden_text': "Email sent to Alex Reed at alex.reed@meridian.com with "
                "the subject 'Meeting Agenda' and the body you "
                "dictated. Done!",
 'required_function_call': {'name': 'send_email',
                            'args': {'to': 'alex.reed@meridian.com'}},
 'function_call_response': {'status': 'success',
                            'message_id': 'MSG-4401'},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_006.wav'},

# ============================================================================
# TURNS 7-18: MID BUCKET — HOTEL + TOPIC SWITCH + CORRECTIONS
# ============================================================================

# Turn 7 — MULTI-INTENT #2: hotel + reminder in one turn
{'input': "I also need a hotel in New York. And oh — can you set a "
          "reminder to pack my bag the night before the trip?",
 'golden_text': "Two things — hotel in New York and a pack-your-bag "
                "reminder. I'll note the reminder and come back to it. "
                "For the hotel — when are you checking in and out, and do "
                "you have a nightly budget in mind?",
 'required_function_call': None,
 'categories': ['basic_qa', 'long_range_memory'],
 'audio_file': 'audio/turn_007.wav'},

# Turn 8 — Hotel dates → list options + user selects
{'input': "Checking in February third, checking out the fifth — two "
          "nights. Nothing too fancy, somewhere in the two-to-three "
          "hundred range.",
 'golden_text': "For two nights, February 3rd to 5th, I have three "
                "options: The Manhattan House at $259 per night in "
                "Midtown, Brooklyn Bridge Suites at $179 per night in "
                "DUMBO, and the Park Avenue Grand at $429 per night on "
                "the Upper East Side. The first two fit your budget — "
                "which sounds best?",
 'required_function_call': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_008.wav'},

# Turn 9 — Select hotel and book
{'input': "The Manhattan House sounds good — standard room, book it "
          "under my name.",
 'golden_text': "Booked! The Manhattan House, standard room, February 3rd "
                "to 5th under Dana Fischer. Your hotel confirmation ID is "
                "HTL-7712.",
 'required_function_call': {'name': 'book_hotel',
                            'args': {'city': 'New York',
                                     'check_in': '2025-02-03',
                                     'check_out': '2025-02-05',
                                     'hotel_name': 'The Manhattan House',
                                     'room_type': 'standard',
                                     'guest_name': 'Dana Fischer'}},
 'function_call_response': {'status': 'success',
                            'confirmation_id': 'HTL-7712',
                            'hotel': 'The Manhattan House',
                            'nightly_rate': 259,
                            'total': 518},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_009.wav'},

# Turn 10 — Handle the reminder from turn 7
{'input': "Now what about that reminder — to pack my bag?",
 'golden_text': "Right — I'll set a reminder for the night before your "
                "trip. That's Sunday, February 2nd. What time would you "
                "like to be reminded?",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_010.wav'},

# Turn 11 — FALSE MEMORY TRAP #1: wrong date (thirteenth vs third)
{'input': "Let's do 8 PM. Wait — the flight was February thirteenth, "
          "right?",
 'golden_text': "Actually, your flight is February 3rd, not the "
                "thirteenth. So the reminder should be February 2nd at "
                "8 PM — want me to go ahead and set that?",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_011.wav'},

# Turn 12 — Confirm reminder + TOPIC SWITCH: email to boss
{'input': "Yes, set it. Also, I need to email my boss Maria about a "
          "different project. Tell her: 'Hi Maria, the Horizon project "
          "deadline is moving to February 14th. I'll send the updated "
          "timeline by Thursday. Best, Dana.'",
 'golden_text': "Done — reminder set for February 2nd at 8 PM. And I've "
                "sent the email to Maria Torres at "
                "maria.torres@meridian.com about the Horizon project.",
 'required_function_call': [
     {'name': 'set_reminder',
     'args': {'date': '2025-02-02',
               'time': '20:00',
               'message': 'Pack bag for NYC trip'}},
     {'name': 'send_email',
      'args': {'to': 'maria.torres@meridian.com'}},
 ],
 'function_call_response': [
     {'status': 'success', 'reminder_id': 'REM-0501'},
     {'status': 'success', 'message_id': 'MSG-4402'},
 ],
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_012.wav'},

# Turn 13 — RETROACTIVE CORRECTION: resend Maria email with correct date
{'input': "Hold on — I said February fourteenth in that email, but the "
          "new deadline is actually the fifteenth. Can you resend it?",
 'golden_text': "Got it — resending to Maria Torres with the corrected "
                "deadline of February 15th instead of the 14th.",
 'required_function_call': {'name': 'send_email',
                            'args': {'to': 'maria.torres@meridian.com'}},
 'function_call_response': {'status': 'success',
                            'message_id': 'MSG-4403'},
 'categories': ['tool_use', 'long_range_memory', 'implicit_correction'],
 'audio_file': 'audio/turn_013.wav'},

# Turn 14 — Add calendar event for return
{'input': "And add a meeting to my calendar for when I get back — "
          "February fifth at 10 AM, one hour, titled 'Team debrief — NYC trip.' "
          "Note that it's in the big conference room.",
 'golden_text': "Added to your calendar — 'Team debrief — NYC trip' on "
                "February 5th at 10:00 AM for one hour, in the big "
                "conference room.",
 'required_function_call': {'name': 'add_calendar_event',
                            'args': {'date': '2025-02-05',
                                     'time': '10:00',
                                     'duration': '1 hour',
                                     'title': 'Team debrief — NYC trip',
                                     'notes': 'Big conference room'}},
 'function_call_response': {'status': 'success',
                            'event_id': 'EVT-8830'},
 'categories': ['tool_use'],
 'audio_file': 'audio/turn_014.wav'},

# Turn 15 — CROSS-REFERENCE: total trip cost
{'input': "So between the flight and the hotel, what am I spending total "
          "on this trip?",
 'golden_text': "Your Delta flight was $315 and The Manhattan House is "
                "$518 for two nights — so your total trip cost is $833.",
 'required_function_call': None,
 'categories': ['long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_015.wav'},

# Turn 16 — AMBIGUOUS ENTITY: "the booking" — flight or hotel?
{'input': "What were the details on the booking again?",
 'golden_text': "Which booking — your Delta flight or The Manhattan House "
                "hotel?",
 'required_function_call': None,
 'categories': ['long_range_memory', 'ambiguous_entity'],
 'audio_file': 'audio/turn_016.wav'},

# Turn 17 — Resolves Turn 16 ambiguity + FALSE MEMORY TRAP #2: wrong email recipient
{'input': "The hotel. And I sent the meeting agenda email to Maria, right?",
 'golden_text': "Your hotel is The Manhattan House — check-in February "
                "3rd, check-out February 5th, $259 per night, two nights, "
                "$518 total. And actually no — the meeting agenda email "
                "was sent to Alex Reed, not Maria. The emails to Maria "
                "were about the Horizon project deadline.",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_017.wav'},

# Turn 18 — FALSE MEMORY TRAP #2b: correction-chain recall
{'input': "And I sent Maria the one with February fourteenth as the "
          "deadline, right?",
 'golden_text': "Not quite — the original email said February 14th, but "
                "you caught the mistake and I resent it with the corrected "
                "deadline of February 15th.",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_018.wav'},

# ============================================================================
# TURNS 19-30: LATE BUCKET — MORE ACTIONS + VAGUE PRONOUN + RECAP + CLOSE
# ============================================================================

# Turn 19 — Check calendar for Feb 3 (should show existing + flight)
{'input': "What's on my calendar for February third now?",
 'golden_text': "On February 3rd you have: the budget review meeting at "
                "8 AM, team standup at 9 AM, and your Delta flight to NYC "
                "at 2:30 PM.",
 'required_function_call': {'name': 'check_calendar',
                            'args': {'date': '2025-02-03'}},
 'function_call_response': {'status': 'success',
                            'events': [
                                {'time': '08:00-09:00',
                                 'title': 'Budget review meeting',
                                 'notes': 'With finance team'},
                                {'time': '09:00-10:00',
                                 'title': 'Team standup',
                                 'notes': 'Weekly recurring'},
                                {'time': '14:30',
                                 'title': 'Flight DL-582 LAX→JFK',
                                 'notes': 'Confirmation FL-3021'}]},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_019.wav'},

# Turn 20 — Another email to Alex about Q4 reports
{'input': "Can you send Alex another email? Say: 'Hi Alex, just a "
          "reminder to bring the Q4 reports to our NYC meeting on "
          "February 3rd. Thanks, Dana.' Subject line 'Q4 Reports for "
          "NYC Meeting.'",
 'golden_text': "Email sent to Alex Reed — 'Q4 Reports for NYC Meeting' "
                "reminding him to bring the Q4 reports.",
 'required_function_call': {'name': 'send_email',
                            'args': {'to': 'alex.reed@meridian.com'}},
 'function_call_response': {'status': 'success',
                            'message_id': 'MSG-4404'},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_020.wav'},

# Turn 21 — CROSS-REFERENCE COUNT: emails sent to Alex
{'input': "How many emails have I sent to Alex today?",
 'golden_text': "You've sent two emails to Alex Reed: the first was 'NYC "
                "Meeting Agenda' about Q4 results and the product roadmap, "
                "and the second was 'Q4 Reports for NYC Meeting' reminding "
                "him to bring the reports.",
 'required_function_call': None,
 'categories': ['long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_021.wav'},

# Turn 22 — Set another reminder (morning of trip)
{'input': "And set a reminder for me on the morning of the third — 7 AM "
          "— to grab my laptop charger.",
 'golden_text': "Reminder set for February 3rd at 7:00 AM — 'Grab laptop "
                "charger.' You're all set.",
 'required_function_call': {'name': 'set_reminder',
                            'args': {'date': '2025-02-03',
                                     'time': '07:00',
                                     'message': 'Grab laptop charger'}},
 'function_call_response': {'status': 'success',
                            'reminder_id': 'REM-0502'},
 'categories': ['tool_use'],
 'audio_file': 'audio/turn_022.wav'},

# Turn 23 — FALSE MEMORY TRAP #3: wrong destination
{'input': "The hotel was in San Francisco, right?",
 'golden_text': "No — your hotel is in New York City, not San Francisco. "
                "The Manhattan House in Midtown Manhattan, February 3rd "
                "to 5th.",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_023.wav'},

# Turn 24 — VAGUE PRONOUN: "forward that to Priya"
{'input': "Can you forward that to Priya too?",
 'golden_text': "Just to confirm — which email would you like forwarded to "
                "Priya? You sent a few today: the NYC meeting agenda, the "
                "Horizon deadline update, or the Q4 reports follow-up.",
 'required_function_call': None,
 'categories': ['long_range_memory', 'ambiguous_entity'],
 'audio_file': 'audio/turn_024.wav'},

# Turn 25 — Resolve pronoun + send to Priya
{'input': "The Horizon one — the corrected version.",
 'golden_text': "Sent the corrected Horizon deadline email to Priya Kapoor "
                "at priya.kapoor@meridian.com.",
 'required_function_call': {'name': 'send_email',
                            'args': {'to': 'priya.kapoor@meridian.com'}},
 'function_call_response': {'status': 'success',
                            'message_id': 'MSG-4405'},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_025.wav'},

# Turn 26 — OVER-CLARIFICATION TRAP: "her" = Priya from immediate context
{'input': "What was her email address again — I want to write it down.",
 'golden_text': "Priya Kapoor's email is priya.kapoor@meridian.com.",
 'required_function_call': None,
 'function_call_response': None,
 'categories': ['ambiguous_entity', 'long_range_memory'],
 'audio_file': 'audio/turn_026.wav'},

# Turn 27 — CROSS-REFERENCE: full recap of all bookings
{'input': "Read me back everything I booked — flights, hotel, the whole "
          "thing.",
 'golden_text': "Here's everything you've booked:\n\n"
                "Flight: Delta DL-582, LAX to JFK, February 3rd at "
                "2:30 PM, one passenger (Dana Fischer), $315. "
                "Confirmation: FL-3021.\n\n"
                "Hotel: The Manhattan House, Midtown Manhattan, standard "
                "room, February 3rd to 5th (2 nights), $259 per night, "
                "$518 total. Confirmation: HTL-7712.\n\n"
                "Total trip cost: $833.\n\n"
                "Does everything look right?",
 'required_function_call': None,
 'categories': ['long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_027.wav'},

# Turn 28 — Recap of emails sent
{'input': "And what emails did I send today?",
 'golden_text': "You sent five emails today:\n\n"
                "1. To Alex Reed — 'NYC Meeting Agenda' about Q4 results "
                "and product roadmap.\n"
                "2. To Maria Torres — 'Horizon Project Deadline Update' "
                "(original, with February 14th).\n"
                "3. To Maria Torres — 'Horizon Project Deadline Update — "
                "Correction' with the corrected date of February 15th.\n"
                "4. To Alex Reed — 'Q4 Reports for NYC Meeting' reminding "
                "him to bring the Q4 reports.\n"
                "5. To Priya Kapoor — 'Horizon Project Deadline Update' "
                "with the February 15th deadline.",
 'required_function_call': None,
 'categories': ['long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_028.wav'},

# Turn 29 — Recap of reminders + modify one (combined multi-intent)
{'input': "And the reminders? Actually — change the pack bag one to "
          "9 PM instead of 8.",
 'golden_text': "You had two reminders:\n\n"
                "1. February 2nd at 8:00 PM — 'Pack bag for NYC trip.'\n"
                "2. February 3rd at 7:00 AM — 'Grab laptop charger.'\n\n"
                "I've updated the first one — pack bag is now set for "
                "February 2nd at 9:00 PM.",
 'required_function_call': {'name': 'set_reminder',
                            'args': {'date': '2025-02-02',
                                     'time': '21:00',
                                     'message': 'Pack bag for NYC trip'}},
 'function_call_response': {'status': 'success',
                            'reminder_id': 'REM-0501',
                            'updated': True},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_029.wav'},

# Turn 30 — Final goodbye + end session
{'input': "Perfect. I think that's everything — thanks so much!",
 'golden_text': "You're all set, Dana! Safe travels to New York on "
                "February 3rd. Have a great meeting!",
 'required_function_call': {'name': 'end_session', 'args': {}},
 'categories': ['tool_use'],
 'audio_file': 'audio/turn_030.wav'},
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
    'numerical_reasoning':  [],

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
