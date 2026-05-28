from crewai import Agent, LLM, Task, Crew, Process

# from langchain_openai import ChatOpenAI

"""
# OpenAI model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)
"""

llm = LLM(
    model="openai/gpt-4o-mini",
    api_key="")

# Define agent
research_agent = Agent(
    role="Research Assistant",
    goal="Provide concise and accurate research summaries",
    backstory=(
        "You are an intelligent AI research assistant "
        "that explains topics clearly."
    ),
    verbose=True,
    llm=llm
)

# Define task
research_task = Task(
    description=(
        "Explain what CrewAI is and why developers use it."
    ),
    expected_output="A short paragraph explanation.",
    agent=research_agent
)

# Create crew
crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    process=Process.sequential,
    verbose=True
)

# Run
result = crew.kickoff()

print("\nFINAL RESULT:\n")
print(result)
