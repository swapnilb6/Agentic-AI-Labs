"""

1. This project demonstrates:

Reinforcement Learning with Human Feedback (RLHF)
Adaptive agentic AI
CrewAI multi-agent orchestration
OpenAI tool/function calling
Human-in-the-loop reward scoring
Dynamic policy tuning
Fintech loan approval workflow

2. Use Case

A fintech company wants AI agents to:

Analyze loan applications
Detect fraud
Check compliance
Make approval decisions
Learn continuously from human reviewer feedback

The system improves over time using RLHF signals.

Flow :

                  ┌──────────────────┐
                  │ Human Reviewer   │
                  │ Reward Feedback  │
                  └────────┬─────────┘
                           │
                           ▼
┌────────────────────────────────────────────┐
│ RLHF Reward Engine                         │
│ - Updates reward scores                    │
│ - Tunes agent behavior                     │
└────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────┐
│ CrewAI Multi-Agent System                  │
│                                            │
│ Financial Agent                            │
│ Fraud Agent                                │
│ Compliance Agent                           │
│ Decision Agent                             │
└────────────────────────────────────────────┘
                           │
                           ▼
                ┌─────────────────┐
                │ OpenAI Tools    │
                │ Function Calls  │
                └─────────────────┘

"""


class AdaptiveMemory:

    def __init__(self):

        self.risk_tolerance = 0.5
        self.fraud_sensitivity = 0.5
        self.compliance_strictness = 0.5

    def update_policy(self, reward_score):

        if reward_score < 0.4:
            self.risk_tolerance -= 0.1
            self.fraud_sensitivity += 0.1
            self.compliance_strictness += 0.1

        elif reward_score > 0.8:
            self.risk_tolerance += 0.05

        self.risk_tolerance = max(0.1, min(1.0, self.risk_tolerance))
        self.fraud_sensitivity = max(0.1, min(1.0, self.fraud_sensitivity))
        self.compliance_strictness = max(
            0.1,
            min(1.0, self.compliance_strictness)
        )

    def get_profile(self):

        return {
            "risk_tolerance": self.risk_tolerance,
            "fraud_sensitivity": self.fraud_sensitivity,
            "compliance_strictness": self.compliance_strictness
        }