CONDENSER_SYSTEM_PROMPT = f"""You are an intelligent and meticulous assistant to a linguisics professor.

You will be presented with a list of features of text.

Some of the features might be essentially the same, just worded differently. Your task is to get rid of these duplicates, resulting in a shorter, condensed list of features.

Only get rid of duplicates if the features are exactly the same as each other, or extremely close.

Do not throw away any details. E.g. if a certain word or string of characters is referenced in a feature, this is key information that must appear in your output.

Get ready to see the list of features."""

CONDENSER_EXAMPLE = """Feature 1: verbs ending in "er".
Feature 2: words ending in "er".
Feature 3: the token "er".
Feature 4: verbs conveying violence.
Feature 5: verbs in the context of violence.
Feature 6: words, ending in "er", especially verbs.
Feature 7: tokens followed by a comma.
Feature 8: "er" at the end of a verb, in the context of sports.
"""
CONDENSER_RESPONSE = """Let me remove the duplicate features. Here is a list of the remaining distinct features:

Feature 1: verbs ending in "er.
Feature 2: words ending in "er".
Feature 3: the token "er".
Feature 4: verbs conveying violence.
Feature 5: tokens followed by a comma.
Feature 6: the token "er" at the end of a verb, in the context of sports."""

def get_condenser_template(explanation_list):
    explanation_str = ''
    for i in range(len(explanation_list)):
        explanation_str += f"Feature {i+1}: {explanation_list[i]}\n"

    template =  f"""
    <|begin_of_text|>

    <|start_header_id|>system<|end_header_id|>
    \n\n{CONDENSER_SYSTEM_PROMPT}
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    \n\n{CONDENSER_EXAMPLE}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    \n\n{CONDENSER_RESPONSE}
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    \n\n{explanation_str}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    \n\n"""

    return template

def get_simple_condenser_template(explanation_list):
    explanation_str = ''
    for i in range(len(explanation_list)):
        explanation_str += f"Feature {i+1}: {explanation_list[i]}\n"

    template =  f"""
    <|begin_of_text|>

    <|start_header_id|>system<|end_header_id|>
    \n\n{CONDENSER_SYSTEM_PROMPT}
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    \n\n{explanation_str}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    \n\n"""

    return template