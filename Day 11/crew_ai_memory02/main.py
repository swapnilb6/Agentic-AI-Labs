"""
refer the code : 

- from human_feedback import get_human_feedback
-   print("\n=== HUMAN FEEDBACK ===")

"""
import os

from crewai import Crew, Process
from openai import OpenAI

from agents import (
    financial_agent,
    fraud_agent,
    compliance_agent,
    decision_agent
)

from tasks import create_tasks

from reward_engine import RewardEngine
from adaptive_memory import AdaptiveMemory

from human_feedback import get_human_feedback

from tools import (
    analyze_credit,
    detect_fraud,
    aml_check
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

memory = AdaptiveMemory()
reward_engine = RewardEngine()


def openai_function_call(customer):

    tools = [
        {
            "type": "function",
            "function": {
                "name": "analyze_credit",
                "description": "Analyze customer credit profile",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "income": {
                            "type": "number"
                        },
                        "debt": {
                            "type": "number"
                        }
                    },
                    "required": ["income", "debt"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "detect_fraud",
                "description": "Detect fraud activity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "unusual_transactions": {
                            "type": "integer"
                        }
                    },
                    "required": ["unusual_transactions"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """
                You are a fintech AI orchestrator.
                Select tools dynamically.
                """
            },
            {
                "role": "user",
                "content": f"""
                Evaluate customer:
                {customer}
                """
            }
        ],
        tools=tools,
        tool_choice="auto"
    )

    tool_calls = response.choices[0].message.tool_calls

    results = []

    for tool_call in tool_calls:

        fn_name = tool_call.function.name

        args = eval(tool_call.function.arguments)

        if fn_name == "analyze_credit":
            result = analyze_credit(args)

        elif fn_name == "detect_fraud":
            result = detect_fraud(args)

        else:
            result = {"error": "Unknown tool"}

        results.append({
            "tool": fn_name,
            "result": result
        })

    return results


def run_rlhf_cycle():

    customer = {
        "income": 100000,
        "debt": 50000,
        "unusual_transactions": 4,
        "country": "HighRiskCountry"
    }

    print("\n=== OPENAI TOOL CALLING ===")

    tool_results = openai_function_call(customer)

    for r in tool_results:
        print(r)

    print("\n=== CREWAI AGENT EXECUTION ===")

    financial = financial_agent()
    fraud = fraud_agent()
    compliance = compliance_agent()
    decision = decision_agent()

    tasks = create_tasks(
        customer,
        financial,
        fraud,
        compliance,
        decision
    )

    crew = Crew(
        agents=[
            financial,
            fraud,
            compliance,
            decision
        ],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    final_decision = crew.kickoff()

    print("\n=== FINAL AI DECISION ===")
    print(final_decision)

    print("\n=== HUMAN FEEDBACK ===")

    feedback = get_human_feedback(str(final_decision))

    print(feedback)

    reward = reward_engine.calculate_reward(feedback)

    print(f"\nReward Score: {reward}")

    print("\n=== POLICY UPDATE ===")

    memory.update_policy(reward)

    print(memory.get_profile())


if __name__ == "__main__":
    run_rlhf_cycle()