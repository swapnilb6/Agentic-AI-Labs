from crewai import Agent

from adaptive_memory import AdaptiveMemory

memory = AdaptiveMemory()


def financial_agent():

    profile = memory.get_profile()

    return Agent(
        role="Financial Risk Analyst",
        goal=f"""
        Evaluate customers using adaptive
        risk tolerance:
        {profile['risk_tolerance']}
        """,
        backstory="""
        Senior fintech lending specialist.
        """,
        verbose=True
    )


def fraud_agent():

    profile = memory.get_profile()

    return Agent(
        role="Fraud Detection Agent",
        goal=f"""
        Detect fraud with sensitivity:
        {profile['fraud_sensitivity']}
        """,
        backstory="""
        AI fraud investigator.
        """,
        verbose=True
    )


def compliance_agent():

    profile = memory.get_profile()

    return Agent(
        role="Compliance Officer",
        goal=f"""
        Apply compliance strictness:
        {profile['compliance_strictness']}
        """,
        backstory="""
        AML and KYC regulatory expert.
        """,
        verbose=True
    )


def decision_agent():

    return Agent(
        role="Loan Decision Agent",
        goal="""
        Produce final loan approval decision
        balancing profitability and safety.
        """,
        backstory="""
        Autonomous fintech decision maker.
        """,
        verbose=True
    )