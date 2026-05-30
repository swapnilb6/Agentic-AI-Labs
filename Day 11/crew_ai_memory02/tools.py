from langsmith import traceable

@traceable
def run_pipeline():
    messages = format_prompt("foo")
    response = invoke_llm(messages)
    return parse_output(response)

def analyze_credit(customer):

    debt_ratio = customer["debt"] / customer["income"]

    if debt_ratio < 0.3:
        return {
            "credit_score": 790,
            "risk_level": "low"
        }

    elif debt_ratio < 0.6:
        return {
            "credit_score": 650,
            "risk_level": "medium"
        }

    return {
        "credit_score": 520,
        "risk_level": "high"
    }


def detect_fraud(customer):

    tx = customer["unusual_transactions"]

    if tx > 5:
        return {
            "fraud_probability": 0.91,
            "flagged": True
        }

    return {
        "fraud_probability": 0.12,
        "flagged": False
    }


def aml_check(customer):

    if customer["country"] in ["HighRiskCountry"]:
        return {
            "aml_risk": "high"
        }

    return {
        "aml_risk": "low"
    }