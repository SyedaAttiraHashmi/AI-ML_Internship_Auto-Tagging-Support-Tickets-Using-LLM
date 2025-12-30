import os
import json
from groq import Groq

class TicketClassifier:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model_name

    def get_prompt(self, ticket_text, mode="zero-shot"):
        
        # Few-shot examples to guide the model's logic
        examples = """
        Example 1:
        Ticket: "I can't log in to my account, it says password incorrect even after reset."
        Output: {"tags": ["Login Issue", "Account Access", "Technical Support"], "justification": "User is experiencing authentication failures."}

        Example 2:
        Ticket: "How do I upgrade my subscription to the Pro plan?"
        Output: {"tags": ["Billing", "Subscription Upgrade", "Sales"], "justification": "Query regarding plan changes and payments."}
        """

        if mode == "few-shot":
            instruction = f"Use the following examples to understand the tagging style:\n{examples}"
        else:
            instruction = "Classify the ticket based on your general knowledge."

        return f"""
        {instruction}
        
        Task: Classify this support ticket into the TOP 3 most probable tags.
        Ticket: {ticket_text}

        Return ONLY a JSON object with:
        "tags": [list of 3 strings],
        "justification": "string"
        """

    def classify(self, ticket_text, mode="zero-shot"):
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": self.get_prompt(ticket_text, mode)}],
                model=self.model,
                temperature=0.1, # Low temperature for consistency
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"tags": ["Error", "Error", "Error"], "justification": str(e)}