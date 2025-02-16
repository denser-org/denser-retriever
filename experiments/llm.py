from openai import OpenAI


class LLM:
    def __init__(self, base_url: str, api_key: str, model: str):
        """Initialize the LLM client.

        Args:
            base_url: Base URL for the OpenAI API
            api_key: Authentication API key
            model: Name of the model to use
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def summarize(self, query: str, output_length: int) -> str:
        """Summarize the input query text.

        Args:
            query: Text to summarize
            output_length: Maximum length of the summary in characters

        Returns:
            Summarized text string
        """
        system_prompt = (
            "You are a precise text summarizer. Follow these rules strictly:\n"
            "1. Produce a summary in no more than {output_length} characters\n"
            "2. Maintain key information\n"
            "3. Use the same language as the input text\n"
            "4. Preserve important named entities, dates, and numbers\n"
            "5. Remove redundant information and filler words\n"
            "6. Do not add any explanations or additional context\n"
            "7. Do not include phrases like 'this text discusses' or 'in summary'\n"
            "8. Keep the original meaning intact while being concise\n"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content
