QUERY_PROMPT = """Thanks for the explanation! Now that you have a hypothesis for what the feature does, I want to know whether this feature is relevant to a neural network's prediction on a specific task. I'll give you a couple task examples plus some pointers for what I'd consider when looking for relevant features.

[TASK EXAMPLES START]
{task_examples}
[TASK EXAMPLES END]

On a scale of 0 to 100, how relevant is this neural feature to the task described above?
- 0 means "completely irrelevant" - this feature does not relate to the task at all. It would not be used in the model's prediction.
- 50 means "somewhat relevant" - this feature could be used in the model's prediction, and it's only somewhat related to the task at hand.
- 100 means "highly relevant" - this feature is important for the model's prediction. It contains important, highly relevant information for the task.

You must answer with a single number between 0 and 100. Don't say anything else, just the number.

Here are a couple things I'd consider when looking for relevant features:
{pointers}"""