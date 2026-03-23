"""
Product comparison benchmark turns (31 turns: 0-30).

Scenario: Ryan Chen calls TechMart Electronics looking for a laptop for his
daughter who's starting college.

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
# TURNS 0-7: EARLY — SETUP + SEARCH + COMPARISON + MULTI-INTENT + RETROACTIVE
# ============================================================================

turns = [
# Turn 0 — Initial request: laptop for college, budget ~$1,300
{'input': "Hi, I'm looking for a laptop for my daughter — she's starting "
          "college in the fall. Budget is around thirteen hundred dollars.",
 'golden_text': "Great! What will she primarily use it for? That'll "
                "help me narrow down the right options.",
 'required_function_call': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_000.wav'},

# Turn 1 — Use case details
{'input': "She's going to be doing a lot of coding — computer science "
          "major — and some graphic design on the side. So she needs "
          "good RAM and decent storage.",
 'golden_text': "For coding and graphic design, I'd recommend at least "
                "16 GB of RAM and 512 GB of storage. Let me search our "
                "catalog.",
 'required_function_call': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_001.wav'},

# Turn 2 — Search products (tool call)
{'input': "Yeah, show me what you've got.",
 'golden_text': "I found four options. The ProBook X1490 at $1,299 with "
                "an Intel i7, 16 GB RAM, and 512 GB storage. The ProBook "
                "X1940 at $1,499 with an i9, 32 GB RAM, and 1 TB "
                "storage. The UltraSlim S15 at $999 with 16 GB RAM but "
                "only 256 GB storage. And the UltraSlim S15 Pro at "
                "$1,199 with 16 GB RAM and 512 GB storage. Which ones "
                "interest you?",
 'required_function_call': {'name': 'search_products',
                            'args': {'use_case': 'coding',
                                     'min_ram': 16}},
 'function_call_response': {'status': 'success',
                            'products': [
                                {'product_id': 'X1490', 'name': 'ProBook X1490',
                                 'price': 1299, 'processor': 'Intel i7-13700H',
                                 'ram': '16GB', 'storage': '512GB', 'gpu': 'Intel Iris Xe'},
                                {'product_id': 'X1940', 'name': 'ProBook X1940',
                                 'price': 1499, 'processor': 'Intel i9-13900H',
                                 'ram': '32GB', 'storage': '1TB', 'gpu': 'NVIDIA RTX 4050'},
                                {'product_id': 'S15', 'name': 'UltraSlim S15',
                                 'price': 999, 'processor': 'AMD Ryzen 7',
                                 'ram': '16GB', 'storage': '256GB', 'gpu': 'AMD Radeon 680M'},
                                {'product_id': 'S15Pro', 'name': 'UltraSlim S15 Pro',
                                 'price': 1199, 'processor': 'AMD Ryzen 9',
                                 'ram': '16GB', 'storage': '512GB', 'gpu': 'AMD Radeon RX 7600S'}]},
 'categories': ['tool_use'],
 'audio_file': 'audio/turn_002.wav'},

# Turn 3 — Model number confusion: X1490 vs X1940
{'input': "The ProBook X-fourteen-ninety — or wait, is it X-nineteen-"
          "forty? Which one's the one with the i7?",
 'golden_text': "The X1490 is the "
                "one with the Intel i7 at $1,299. The X1940 "
                "has the Intel i9 at $1,499. Easy to "
                "mix up! Want me to compare them side by side?",
 'required_function_call': None,
 'categories': ['ambiguous_entity'],
 'audio_file': 'audio/turn_003.wav'},

# Turn 4 — Compare X1490 vs X1940 (tool call)
{'input': "Yeah, compare those two ProBooks for me.",
 'golden_text': "Here's the side-by-side: The X1490 has an i7 processor, "
                "16 GB RAM, 512 GB storage, and Intel Iris Xe graphics "
                "at $1,299. The X1940 has an i9 processor, 32 GB RAM, "
                "1 TB storage, and an NVIDIA RTX 4050 dedicated GPU at "
                "$1,499. The X1940 is significantly more powerful, "
                "especially for graphic design with that dedicated GPU, "
                "but it's $200 more.",
 'required_function_call': {'name': 'compare_specs',
                            'args': {'product_ids': ['X1490', 'X1940']}},
 'function_call_response': {'status': 'success',
                            'comparison': {
                                'X1490': {'processor': 'Intel i7-13700H', 'ram': '16GB',
                                          'storage': '512GB', 'gpu': 'Intel Iris Xe',
                                          'battery': '10 hours', 'weight': '4.2 lbs',
                                          'price': 1299},
                                'X1940': {'processor': 'Intel i9-13900H', 'ram': '32GB',
                                          'storage': '1TB', 'gpu': 'NVIDIA RTX 4050',
                                          'battery': '8 hours', 'weight': '4.8 lbs',
                                          'price': 1499}}},
 'categories': ['tool_use'],
 'audio_file': 'audio/turn_004.wav'},

# Turn 5 — RAM question + KB grounding
{'input': "Is sixteen gigs of RAM going to be enough for coding and "
          "graphic design? Or does she really need thirty-two?",
 'golden_text': "For coding and software development, 16 GB is "
                "recommended — our catalog lists 16+ GB for that use "
                "case. 32 GB is recommended for video editing and 3D "
                "rendering. Since she's mainly coding with some graphic "
                "design, 16 GB should be a good fit.",
 'required_function_call': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_005.wav'},

# Turn 6 — MULTI-INTENT: S15 Pro comparison + charger question
{'input': "What about the UltraSlim S15 Pro? How does that compare to "
          "the X1490? And do any of these come with a USB-C charger in "
          "the box?",
 'golden_text': "The S15 Pro is $100 cheaper at $1,199, has the same "
                "16 GB RAM and 512 GB storage, but with an AMD Ryzen 9 "
                "instead of Intel i7, and it has a dedicated AMD Radeon "
                "RX 7600S GPU. It also has a higher-res QHD display "
                "versus the X1490's Full HD. The S15 Pro is lighter at "
                "3.8 pounds versus 4.2, but gets 9 hours of battery "
                "versus 10. As for what's included in the box, I don't "
                "have those details in our product specs. I'd recommend "
                "checking the product page on our website or I can have "
                "someone follow up on that.",
 'required_function_call': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_006.wav'},

# Turn 7 — RETROACTIVE CORRECTION via reported speech
{'input': "Hold on — my daughter's looking over my shoulder. She says "
          "it's actually going to be more video editing and 3D modeling "
          "than graphic design — she's doing a digital media minor. "
          "And she says she really cares about weight and battery since "
          "she'll be carrying it around campus all day. Does that change "
          "things?",
 'golden_text': "That does change things. For video editing and 3D "
                "modeling, 32 GB of RAM and a dedicated GPU become much "
                "more important — that points to the X1940 with its "
                "RTX 4050 and 32 GB. However, the X1940 is the heaviest "
                "at 4.8 pounds with 8 hours of battery. If portability "
                "matters a lot, the S15 Pro is a compromise — 3.8 pounds, "
                "9 hours of battery, and it has a dedicated GPU, but only "
                "16 GB of RAM. For serious video editing and 3D, the "
                "X1940 is the better fit despite the weight trade-off.",
 'required_function_call': None,
 'categories': ['long_range_memory', 'implicit_correction'],
 'audio_file': 'audio/turn_007.wav'},

# ============================================================================
# TURNS 8-19: MID — BUDGET MATH + ARITHMETIC + POLICY EDGE + TRAPS
# ============================================================================

# Turn 8 — Compare S15 Pro vs X1490 (tool call)
{'input': "Can you pull up a side-by-side of the S15 Pro and the X1490?",
 'golden_text': "Here's the comparison: The X1490 has an Intel i7, "
                "16 GB RAM, 512 GB storage, Intel Iris Xe graphics, "
                "Full HD display, 10 hours battery, 4.2 lbs, at $1,299. "
                "The S15 Pro has an AMD Ryzen 9, 16 GB RAM, 512 GB "
                "storage, AMD Radeon RX 7600S dedicated GPU, QHD "
                "display, 9 hours battery, 3.8 lbs, at $1,199.",
 'required_function_call': {'name': 'compare_specs',
                            'args': {'product_ids': ['X1490', 'S15Pro']}},
 'function_call_response': {'status': 'success',
                            'comparison': {
                                'X1490': {'processor': 'Intel i7-13700H', 'ram': '16GB',
                                          'storage': '512GB', 'gpu': 'Intel Iris Xe',
                                          'display': '15.6" FHD', 'battery': '10 hours',
                                          'weight': '4.2 lbs', 'price': 1299},
                                'S15Pro': {'processor': 'AMD Ryzen 9 7940HS', 'ram': '16GB',
                                           'storage': '512GB', 'gpu': 'AMD Radeon RX 7600S',
                                           'display': '15.6" QHD', 'battery': '9 hours',
                                           'weight': '3.8 lbs', 'price': 1199}}},
 'categories': ['tool_use'],
 'audio_file': 'audio/turn_008.wav'},

# Turn 9 — VAGUE PRONOUN: "that one" after multiple products discussed
{'input': "What was the battery life on that one again?",
 'golden_text': "Just to clarify — which model are you asking about? "
                "We've been looking at a few: the X1490, the X1940, "
                "and the S15 Pro.",
 'required_function_call': None,
 'categories': ['ambiguous_entity'],
 'audio_file': 'audio/turn_009.wav'},

# Turn 10 — MERGED: resolve pronoun + daughter wants X1940 over budget
{'input': "The S15 Pro — the lightweight one. But honestly, my daughter "
          "just looked at the specs and she really wants the X1940 "
          "with the thirty-two gigs and the NVIDIA card. Problem is, "
          "that's fourteen ninety-nine, which is about two hundred over "
          "my thirteen hundred budget.",
 'golden_text': "The S15 Pro gets up to 9 hours of battery. As for the "
                "X1940 — it is $1,499 new, which is $199 over your "
                "$1,300 budget. But there are two ways to bring the "
                "price down: an open-box model at $1,199 — that's "
                "within your budget — or a 10 percent student discount "
                "with a .edu email, which brings it to about $1,349. "
                "Want to explore either option?",
 'required_function_call': None,
 'categories': ['long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_010.wav'},

# Turn 11 — Check student discount on X1940 (tool call)
{'input': "Yeah, she'll have a .edu email. Check the student price.",
 'golden_text': "With the 10 percent student discount, the X1940 comes "
                "down to $1,349.10. That's still about $49 over your "
                "$1,300 budget, but much closer.",
 'required_function_call': {'name': 'check_student_discount',
                            'args': {'product_id': 'X1940'}},
 'function_call_response': {'status': 'success',
                            'original_price': 1499,
                            'discount_percent': 10,
                            'discounted_price': 1349.10},
 'categories': ['tool_use', 'numerical_reasoning'],
 'audio_file': 'audio/turn_011.wav'},

# Turn 12 — CONDITIONAL ARITHMETIC CHAIN: compare two full configurations
{'input': "Hmm, still over. Let me think about this differently. "
          "What if we did the S15 Pro with the student discount, the "
          "three-year extended warranty, and express shipping — what's "
          "the total? And compare that to the open-box X1940 with just "
          "accidental damage protection and free standard shipping.",
 'golden_text': "Let me run those numbers. The S15 Pro with student "
                "discount is $1,079.10, plus the 3-year extended "
                "warranty at $149, plus express shipping at $19.99 — "
                "that's $1,248.09 total. The open-box X1940 is $1,199, "
                "plus accidental damage protection at $79 per year, "
                "with free standard shipping — that's $1,278 total. "
                "So the S15 Pro config is about $30 cheaper, but the "
                "X1940 gets you 32 GB RAM and the RTX 4050 — much "
                "better for video editing and 3D.",
 'required_function_call': None,
 'categories': ['numerical_reasoning', 'long_range_memory'],
 'audio_file': 'audio/turn_012.wav'},

# Turn 13 — DISCOUNT STACKING / POLICY EDGE
{'input': "What about the student discount on the open-box X1940? "
          "Can we stack those?",
 'golden_text': "Unfortunately, the student discount doesn't stack with "
                "open-box pricing. It's one or the other. But the "
                "open-box X1940 at $1,199 is already $300 off the new "
                "price — a bigger savings than the 10 percent student "
                "discount would give you on the new model.",
 'required_function_call': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_013.wav'},

# Turn 14 — SUBTLE FALSE TRAP #1: wrong GPU embedded in casual sentence
{'input': "OK so the open-box X1940 with the RTX 4060 and the terabyte "
          "drive — that's eleven ninety-nine, right?",
 'golden_text': "Small correction — the X1940 has the RTX 4050, not "
                "the 4060. But yes, it has the 1 TB drive and the "
                "open-box price is $1,199.",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_014.wav'},

# Turn 15 — SUBTLE FALSE TRAP #2: wrong storage stated as fact
{'input': "Got it, 4050. Between the X1490's 256-gig drive and the "
          "X1940's terabyte, that's a big jump in storage.",
 'golden_text': "Actually, the X1490 has a 512 GB drive, not 256. "
                "You might be thinking of the UltraSlim S15 — that's "
                "the one with 256 GB. So it's 512 GB on the X1490 "
                "versus 1 TB on the X1940 — still a meaningful "
                "difference, but not as dramatic.",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_015.wav'},

# Turn 16 — S15 Pro student discount (tool call)
{'input': "Right, 512. OK just for reference — what would the S15 Pro "
          "be with the student discount?",
 'golden_text': "The S15 Pro is $1,199 new, so with the 10 percent "
                "student discount, it would be $1,079.10. That's the "
                "most affordable option with a dedicated GPU and 512 GB "
                "storage.",
 'required_function_call': {'name': 'check_student_discount',
                            'args': {'product_id': 'S15Pro'}},
 'function_call_response': {'status': 'success',
                            'original_price': 1199,
                            'discount_percent': 10,
                            'discounted_price': 1079.10},
 'categories': ['tool_use', 'numerical_reasoning'],
 'audio_file': 'audio/turn_016.wav'},

# Turn 17 — MERGED: decision (open-box X1940) + SUBTLE FALSE TRAP #3 (QHD)
{'input': "You know what, let's go with the open-box X1940 at eleven "
          "ninety-nine. She gets the same QHD display as the S15 Pro "
          "but with thirty-two gigs of RAM — best of both worlds.",
 'golden_text': "Great choice on the X1940! One thing though — the "
                "X1940 actually has a Full HD display, not QHD. The "
                "S15 Pro is the one with the QHD screen. But the X1940 "
                "makes up for it with the 32 GB RAM, the i9 processor, "
                "1 TB storage, and the RTX 4050 — all excellent for "
                "video editing and 3D work. And at $1,199 open-box, "
                "it's $101 under your $1,300 budget before any add-ons.",
 'required_function_call': None,
 'categories': ['long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_017.wav'},

# Turn 18 — CROSS-REFERENCE: which products have dedicated GPUs
{'input': "Out of the four laptops you showed me, which ones actually "
          "have a dedicated GPU?",
 'golden_text': "Two of the four have dedicated GPUs: the ProBook X1940 "
                "with the NVIDIA RTX 4050, and the UltraSlim S15 Pro "
                "with the AMD Radeon RX 7600S. The X1490 and the "
                "regular S15 both have integrated graphics only.",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_018.wav'},

# Turn 19 — FALSE TRAP #4: wrong RAM for S15 Pro
{'input': "And the S15 Pro had thirty-two gigs of RAM too, right?",
 'golden_text': "No — the S15 Pro has 16 GB of RAM. The X1940 is the "
                "only model with 32 GB. You're making the right call "
                "if she needs 32 GB for video editing and 3D.",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_019.wav'},

# ============================================================================
# TURNS 20-30: LATE — WARRANTY + CART + MODS + OUT-OF-SCOPE + RECAP + CLOSE
# ============================================================================

# Turn 20 — MERGED: warranty KB question + pick warranty + order details
{'input': "Good call on the X1940 then. Does the open-box come with a "
          "full warranty? Actually — add the accidental damage "
          "protection. Name for the order is Ryan Chen — C-H-E-N. "
          "She'll want it shipped to school.",
 'golden_text': "Yes, open-box products come with the full 1-year "
                "standard warranty, same as new. I'll add accidental "
                "damage protection at $79 per year — covers drops, "
                "spills, and cracked screens. Great for a college "
                "student. Got it — Ryan Chen, C-H-E-N. I'll just need "
                "a good phone number for the order.",
 'required_function_call': None,
 'categories': ['basic_qa', 'long_range_memory'],
 'audio_file': 'audio/turn_020.wav'},

# Turn 21 — Phone number + add to cart (tool call)
{'input': "201-473-1560.",
 'golden_text': "Added to your cart: ProBook X1940 open-box at $1,199, "
                "with accidental damage protection at $79. Your total "
                "is $1,278. Cart ID is CART-5501.",
 'required_function_call': {'name': 'add_to_cart',
                            'args': {'product_id': 'X1940',
                                     'customer_name': 'Ryan Chen',
                                     'phone': '201-473-1560',
                                     'condition': 'open_box',
                                     'warranty': 'accidental_damage',
                                     'student_discount': False}},
 'function_call_response': {'status': 'success',
                            'cart_id': 'CART-5501',
                            'product': 'ProBook X1940 (open-box)',
                            'product_price': 1199,
                            'warranty_cost': 79,
                            'total': 1278},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_021.wav'},

# Turn 22 — OUT-OF-SCOPE: iPhone question
{'input': "Great. Oh — my son also needs a phone for school. Do you "
          "carry the iPhone 16?",
 'golden_text': "I'm the sales assistant for TechMart Electronics — "
                "we specialize in laptops and computers, so I can't "
                "help with phones. Is there anything else I can do "
                "for the laptop order?",
 'required_function_call': None,
 'categories': ['out_of_scope'],
 'audio_file': 'audio/turn_022.wav'},

# Turn 23 — MERGED: shipping question + pick express (tool call)
{'input': "No worries. What are the shipping options? Just go with "
          "express — that's the two to three day one, right?",
 'golden_text': "Yes — express shipping is $19.99 and takes 2 to 3 "
                "business days. We also have free standard in 5 to 7 days "
                "and next-day for $34.99. I've updated your cart to express "
                "shipping, and your new total is $1,297.99.",
 'required_function_call': {'name': 'update_cart',
                            'args': {'cart_id': 'CART-5501',
                                     'field': 'shipping',
                                     'new_value': 'express'}},
 'function_call_response': {'status': 'success',
                            'shipping_cost': 19.99,
                            'new_total': 1297.99},
 'categories': ['basic_qa', 'tool_use'],
 'audio_file': 'audio/turn_023.wav'},

# Turn 24 — ORDER MODIFICATION #1: switch warranty to extended
{'input': "Actually, switch to the three-year extended warranty instead "
          "of the accidental damage plan.",
 'golden_text': "Updated — switching from accidental damage protection "
                "at $79 to the 3-year extended warranty at $149. Your "
                "new total is $1,367.99.",
 'required_function_call': {'name': 'update_cart',
                            'args': {'cart_id': 'CART-5501',
                                     'field': 'warranty',
                                     'new_value': 'extended'}},
 'function_call_response': {'status': 'success',
                            'old_warranty': 'accidental_damage',
                            'new_warranty': 'extended',
                            'warranty_cost': 149,
                            'new_total': 1367.99},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_024.wav'},

# Turn 25 — ORDER MODIFICATION #2: change shipping to next-day
{'input': "And change shipping to next-day — I want it there by Monday.",
 'golden_text': "Updated — next-day shipping at $34.99 instead of "
                "express at $19.99. New total is $1,382.99.",
 'required_function_call': {'name': 'update_cart',
                            'args': {'cart_id': 'CART-5501',
                                     'field': 'shipping',
                                     'new_value': 'next_day'}},
 'function_call_response': {'status': 'success',
                            'old_shipping': 'express',
                            'new_shipping': 'next_day',
                            'shipping_cost': 34.99,
                            'new_total': 1382.99},
 'categories': ['tool_use'],
 'audio_file': 'audio/turn_025.wav'},

# Turn 26 — ORDER MODIFICATION #3: update phone number
{'input': "Oh wait — update my phone number. It's 201-473-1565, "
          "not 1560.",
 'golden_text': "Updated — phone number changed from 201-473-1560 to "
                "201-473-1565. Everything else stays the same.",
 'required_function_call': {'name': 'update_cart',
                            'args': {'cart_id': 'CART-5501',
                                     'field': 'phone',
                                     'new_value': '201-473-1565'}},
 'function_call_response': {'status': 'success',
                            'field': 'phone',
                            'old_value': '201-473-1560',
                            'new_value': '201-473-1565'},
 'categories': ['tool_use', 'long_range_memory'],
 'audio_file': 'audio/turn_026.wav'},

# Turn 27 — Price clarification: new vs open-box pricing
{'input': "So this is the fourteen ninety-nine model, right?",
 'golden_text': "The X1940 is $1,499 new, but you're getting the "
                "open-box version at $1,199. Your total with the "
                "3-year extended warranty and next-day shipping is "
                "$1,382.99.",
 'required_function_call': None,
 'categories': ['long_range_memory', 'numerical_reasoning'],
 'audio_file': 'audio/turn_027.wav'},

# Turn 28 — ENHANCED RECAP: must reflect all modifications
{'input': "Right, open-box. Can you read me back the whole order with "
          "all those changes?",
 'golden_text': "Here's your order:\n\n"
                "Customer: Ryan Chen\n"
                "Phone: 201-473-1565\n"
                "Product: ProBook X1940 (open-box) — $1,199\n"
                "Specs: Intel i9, 32GB RAM, 1TB storage, RTX 4050\n"
                "Warranty: 3-year extended warranty — $149\n"
                "Shipping: Next-day — $34.99\n"
                "Total: $1,382.99\n"
                "Cart ID: CART-5501\n\n"
                "Everything look right?",
 'required_function_call': None,
 'categories': ['long_range_memory'],
 'audio_file': 'audio/turn_028.wav'},

# Turn 29 — MULTI-INTENT: return policy + financing question
{'input': "Perfect. What's the return policy on open-box? And what if "
          "I want to finance this instead of paying all at once?",
 'golden_text': "Open-box products have a 15-day return window with a "
                "full refund, no restocking fee. If there's any defect, "
                "return shipping is free. For financing, we offer 0% "
                "APR for 12 months on orders over $500 through TechMart "
                "Credit — so this order would qualify.",
 'required_function_call': None,
 'categories': ['basic_qa'],
 'audio_file': 'audio/turn_029.wav'},

# Turn 30 — Final goodbye + end session
{'input': "That all sounds good. I think I'm set for now. Thanks for "
          "all the help!",
 'golden_text': "You're all set with the ProBook X1940 in your cart "
                "with the 3-year extended warranty and next-day shipping. "
                "Thanks for calling TechMart, and have a great day!",
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
    'out_of_scope':         [],
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
