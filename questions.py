# questions.py
from typing import Dict, List

QUESTIONS: Dict[str, List[str]] = {
    "requires_license": [
        "Quote the sentence that requires a business licence to operate or advertise a short-term rental.",
        "Find the sentence stating a business licence is required to operate a short-term rental.",
        "What sentence says a valid business licence must be held for short-term rental activity?"
    ],
    "principal_residence_only": [
        "Quote the sentence that restricts short-term rental to a principal residence.",
        "Find the sentence stating short-term rentals may only be provided in a principal residence.",
        "What sentence limits short-term rental to the operator's principal residence?",
        "What sentence says short-term rentals are only permitted in your principal dwelling unit?"
    ],
    "display_license_on_listing": [
        "Quote the sentence that requires the business licence number to be included in any advertising or listing.",
        "Find the sentence stating the business licence number must appear in listings or advertisements.",
        "Which sentence mandates including the licence number on listings?"
    ],
    "provincial_registration_required": [
        "Do operators have to register with the Province’s short-term rental registry?",
        "Is a provincial short-term rental registration ID required for operators?",
        "Must listings display a provincial short-term rental registration number?",
        "Quote any sentence that requires a provincial short-term rental registration number to appear in listings.",
        "Which sentence requires provincial registration in addition to the city licence?"
    ],
    "max_entire_home_nights": [
        "What is the maximum number of nights per calendar year an entire home may be rented as a short-term rental?",
        "How many nights can an entire dwelling be rented annually?",
        "Is there a yearly cap on entire-home short-term rental nights?",
        "When away, what is the maximum number of nights per calendar year an entire unit may be rented?",
        "Is there a yearly night cap for entire-home short-term rentals (for example, 160 nights)?",
        "Quote the sentence that states the maximum number of nights per year an entire dwelling may be rented as a short-term rental."
    ]
}
