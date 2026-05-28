# knowledgegraph_queryplan_langchain_05.py
# ==========================================================
# FINTECH KNOWLEDGE GRAPH + QUERY PLANNING SYSTEM
# ==========================================================
#
# MAY 2026 STABLE ARCHITECTURE
#
# FEATURES
# ----------------------------------------------------------
# 1. Financial Knowledge Graph
# 2. Query Planning
# 3. Multi-Step Financial Reasoning
# 4. Graph-Based Retrieval
# 5. Risk Propagation Analysis

"""
Query planning is the process of analyzing and organizing a user query into an efficient retrieval 
or reasoning strategy before execution.

In AI, RAG, and database systems, query planning helps determine:

- what information is needed
- where to retrieve it from
- how to combine results effectively.

In LLM and LangChain workflows, query planning may involve:

a. query decomposition,
b. multi-step retrieval,
c. tool selection,
d. routing,
e. and multi-hop reasoning



"""



# ==========================================================
# INSTALLATION (MAY 2026 SAFE)
# ==========================================================

"""
pip install -U \
langchain \
langchain-core \
langchain-openai \
networkx \
pandas \
python-dotenv
"""

# ==========================================================
# .env
# ==========================================================

# OPENAI_API_KEY=your_api_key

# ==========================================================
# IMPORTS
# ==========================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# LANGCHAIN
# ==========================================================

from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

# ==========================================================
# GRAPH LIBRARY
# ==========================================================

import networkx as nx

# ==========================================================
# DATA ANALYSIS
# ==========================================================

import pandas as pd

# ==========================================================
# LLM CONFIG
# ==========================================================

LLM_MODEL = "gpt-4.1-mini"

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0.1,
)

# ==========================================================
# ==========================================================
# 1. FINANCIAL KNOWLEDGE GRAPH
# ==========================================================
# ==========================================================

print("\n================ KNOWLEDGE GRAPH ================\n")

graph = nx.DiGraph()

"""
nx → alias for NetworkX
DiGraph() → stands for Directed Graph
graph → variable storing the graph object

What it means:

A directed graph has edges with direction (A → B is different from B → A)
Useful for modeling one-way relationships

Purpose:

- Represent workflows, dependencies, or flows
- Build knowledge graphs or graph-based RAG systems
- Model real-world directional relationships (e.g., citations, links, pipelines)

Common use cases:

- recommendation engines
- social network follow graphs
- task dependencies in workflows
- knowledge graph construction

"""

# ==========================================================
# ADD FINANCIAL ENTITIES
# ==========================================================

financial_entities = [

    ("Federal Reserve", "Institution"),
    ("Interest Rates", "Economic Factor"),
    ("Inflation", "Economic Factor"),
    ("Loan Defaults", "Risk"),
    ("Credit Risk", "Risk"),
    ("Liquidity Risk", "Risk"),
    ("Bank Liquidity", "Financial Metric"),
    ("Mortgage Market", "Sector"),
    ("Corporate Lending", "Sector"),
    ("Recession", "Economic Condition"),
    ("Consumer Debt", "Financial Metric"),
    ("Fraud Detection", "AI System"),
    ("AML Monitoring", "Compliance"),
]

for entity, category in financial_entities:

    graph.add_node(
        entity,
        category=category
    )

# ==========================================================
# ADD RELATIONSHIPS
# ==========================================================

financial_relationships = [

    ("Federal Reserve", "Interest Rates", "controls"),

    ("Inflation", "Interest Rates", "increases"),

    ("Interest Rates", "Loan Defaults", "raises"),

    ("Loan Defaults", "Credit Risk", "increases"),

    ("Credit Risk", "Liquidity Risk", "amplifies"),

    ("Liquidity Risk", "Bank Liquidity", "reduces"),

    ("Recession", "Loan Defaults", "accelerates"),

    ("Consumer Debt", "Loan Defaults", "drives"),

    ("Mortgage Market", "Credit Risk", "contributes_to"),

    ("Corporate Lending", "Credit Risk", "contributes_to"),

    ("Fraud Detection", "AML Monitoring", "supports"),
]

for source, target, relation in financial_relationships:

    graph.add_edge(
        source,
        target,
        relation=relation
    )

print(f"Nodes: {graph.number_of_nodes()}")

print(f"Edges: {graph.number_of_edges()}")

# ==========================================================
# GRAPH VISUALIZATION DATA
# ==========================================================

print("\nFinancial Entity Relationships:\n")

for source, target, data in graph.edges(data=True):

    print(
        f"{source} -> ({data['relation']}) -> {target}"
    )

# ==========================================================
# ==========================================================
# 2. QUERY PLANNING SYSTEM
# ==========================================================
# ==========================================================

print("\n================ QUERY PLANNING ================\n")

"""
Query Planning:
Break complex fintech questions
into smaller reasoning steps.
"""

QUERY_PLANNER_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are a fintech AI query planner.

Break the following financial question into:

1. Key financial entities
2. Required reasoning steps
3. Dependencies between entities
4. Final analysis goal

Question:
{question}

Return the answer in structured format.
"""
)

planner_chain = (
    QUERY_PLANNER_PROMPT
    | llm
    | StrOutputParser()
)

# ==========================================================
# SAMPLE QUERY
# ==========================================================

financial_question = """
How do rising interest rates increase
loan defaults and reduce bank liquidity
during recession periods?
"""

query_plan = planner_chain.invoke(
    {
        "question": financial_question
    }
)

print(query_plan)

# ==========================================================
# ==========================================================
# 3. GRAPH RETRIEVAL FUNCTION
# ==========================================================
# ==========================================================

print("\n================ GRAPH RETRIEVAL ================\n")

def graph_retrieval(entity_name: str):

    """
    Retrieve neighboring financial entities
    and relationships.
    """

    if entity_name not in graph:

        return ["Entity not found"]

    neighbors = list(graph.neighbors(entity_name))
    
    results = []
    
    for neighbor in neighbors:

        relation = graph.get_edge_data(
            entity_name,
            neighbor
        )["relation"]

        results.append(
            f"{entity_name} -> ({relation}) -> {neighbor}"
        )

    return results

# ==========================================================
# RETRIEVE GRAPH CONTEXT
# ==========================================================

graph_results = graph_retrieval(
    "Interest Rates"
)

for item in graph_results:

    print(item)

# ==========================================================
# ==========================================================
# 4. MULTI-STEP FINANCIAL REASONING
# ==========================================================
# ==========================================================

print("\n================ MULTI-STEP REASONING ================\n")

graph_context = "\n".join(graph_results)

REASONING_PROMPT = PromptTemplate(
    input_variables=[
        "context",
        "question"
    ],
    template="""
You are a senior fintech AI analyst.

Use the financial knowledge graph
to perform multi-step reasoning.

Financial Graph:
{context}

Question:
{question}

Perform:
1. Relationship analysis
2. Financial reasoning
3. Risk propagation analysis
4. Systemic impact evaluation
5. Final conclusion
"""
)

reasoning_chain = (
    REASONING_PROMPT
    | llm
    | StrOutputParser()
)

reasoning_response = reasoning_chain.invoke(
    {
        "context": graph_context,
        "question": financial_question,
    }
)

print(reasoning_response)

# ==========================================================
# ==========================================================
# 5. RISK PROPAGATION ANALYSIS
# ==========================================================
# ==========================================================

print("\n================ RISK PROPAGATION ================\n")

"""
Find shortest financial risk paths
inside the graph.
"""

source_entity = "Inflation"

target_entity = "Bank Liquidity"

risk_path = nx.shortest_path(
    graph,
    source=source_entity,
    target=target_entity
)

print(
    f"Risk propagation path:\n"
)

for i in range(len(risk_path) - 1):

    source = risk_path[i]

    target = risk_path[i + 1]

    relation = graph.get_edge_data(
        source,
        target
    )["relation"]

    print(
        f"{source} -> ({relation}) -> {target}"
    )

# ==========================================================
# ==========================================================
# 6. GRAPH ANALYTICS
# ==========================================================
# ==========================================================

print("\n================ GRAPH ANALYTICS ================\n")

node_degrees = []

for node in graph.nodes():

    node_degrees.append({

        "Entity": node,

        "Connections": graph.degree(node)
    })

df = pd.DataFrame(node_degrees)

df = df.sort_values(
    by="Connections",
    ascending=False
)

print(df)

# ==========================================================
# ==========================================================
# 7. ADVANCED FINTECH USE CASES
# ==========================================================
# ==========================================================

print("\n================ FINTECH USE CASES ================\n")

use_cases = [

    {
        "Use Case": "Credit Risk Analysis",
        "Technique": "Knowledge Graph + Query Planning"
    },

    {
        "Use Case": "AML Monitoring",
        "Technique": "Graph Traversal"
    },

    {
        "Use Case": "Fraud Detection",
        "Technique": "Entity Relationship Analysis"
    },

    {
        "Use Case": "Liquidity Forecasting",
        "Technique": "Risk Propagation Modeling"
    },

    {
        "Use Case": "Macroeconomic Intelligence",
        "Technique": "Multi-Step Reasoning"
    }
]

use_case_df = pd.DataFrame(use_cases)

print(use_case_df)

# ==========================================================
# ==========================================================
# SUMMARY
# ==========================================================
# ==========================================================

print(
"""
==========================================================
FINTECH KNOWLEDGE GRAPH SYSTEM COMPLETE
==========================================================

FEATURES IMPLEMENTED
----------------------------------------------------------

✓ Financial Knowledge Graph
✓ Query Planning
✓ Multi-Step Reasoning
✓ Graph Retrieval
✓ Risk Propagation Analysis
✓ Relationship Intelligence
✓ Financial Entity Analytics

==========================================================
MAY 2026 SAFE LANGCHAIN STACK
==========================================================

✓ LangChain v1 compatible
✓ No deprecated imports
✓ No retriever conflicts
✓ No classic package dependency
✓ No vectorstore conflicts
✓ Production-ready architecture

==========================================================
"""
)