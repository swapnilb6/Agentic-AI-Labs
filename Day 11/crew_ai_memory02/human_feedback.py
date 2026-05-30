"""
feedback = {
    "correct_decision": True,
    "compliant": True,
    "fraud_detected": True,
    "safe_reasoning": True
}
"""

def get_human_feedback(ai_decision):

    """
    Simulated RLHF reviewer
    """

    if "reject" in ai_decision.lower():

        return {
            "correct_decision": True,
            "compliant": True,
            "fraud_detected": True,
            "safe_reasoning": True
        }

    return {
        "correct_decision": False,
        "compliant": True,
        "fraud_detected": False,
        "safe_reasoning": True
    }