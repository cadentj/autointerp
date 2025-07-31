QUERY_PROMPT = """Now that you have a hypothesis for what the feature does, please assign the feature a relevance score according to the following rubric: 

<rubric>
{pointers}
</rubric>

The relevance score should be a number between 0 and 100:
* 0 means the feature shows no meaningful activation patterns related to the target concept
* 50 means the feature occasionally activates on the target concept but with significant inconsistencies or secondary activations
* 100 means the feature consistently and primarily activates on the target concept

Consider both the feature's activation patterns and the hypothesis you provided.

You must answer with a single number between 0 and 100. Don't say anything else, just the number."""