class ConversationMemory:
    def __init__(self):
        self.messages = []

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        """
        self.messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    def get_messages(self) -> list[dict]:
        """
        Return the conversation history.
        """
        return self.messages.copy()

    def clear(self) -> None:
        """
        Clear the conversation history.
        """
        self.messages.clear()