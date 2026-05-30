from crewai import Task


def create_tasks(
    customer,
    financial,
    fraud,
    compliance,
    decision
):

    t1 = Task(
        description=f"""
        Analyze customer financial profile:

        {customer}
        """,
        expected_output="Financial assessment",
        agent=financial
    )

    t2 = Task(
        description=f"""
        Detect fraud indicators:

        {customer}
        """,
        expected_output="Fraud analysis",
        agent=fraud
    )

    t3 = Task(
        description=f"""
        Perform AML and compliance review.
        """,
        expected_output="Compliance report",
        agent=compliance
    )

    t4 = Task(
        description="""
        Produce final approve/reject decision.
        """,
        expected_output="Final lending decision",
        agent=decision
    )

    return [t1, t2, t3, t4]