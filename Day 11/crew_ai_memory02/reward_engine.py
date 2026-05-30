class RewardEngine:

    def calculate_reward(self, human_feedback):

        """
        Reward score:
        1.0 = excellent
        0.0 = poor
        """

        score = 0

        if human_feedback["correct_decision"]:
            score += 0.4

        if human_feedback["compliant"]:
            score += 0.3

        if human_feedback["fraud_detected"]:
            score += 0.2

        if human_feedback["safe_reasoning"]:
            score += 0.1

        return round(score, 2)