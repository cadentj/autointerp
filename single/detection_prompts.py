DSCORER_SYSTEM_PROMPT = """You are an intelligent and meticulous linguistics researcher.

You will be given a certain feature of text, such as "male pronouns" or "text with negative sentiment".

You will then be given several text examples. Some examples possess the feature, but some do not. Your task is to determine which examples possess the feature.

The last line of your answer must have the form FINAL ANSWER: [list of numbers]
"""


DSCORER_EXAMPLE = """Feature explanation: male pronouns and names.

Text examples:

Example 1: "unfortunately", but we knew that later, a great deal of money would
Example 2: of the president, but after the process had stopped, he stated that
Example 3: Reuters (2017) 27th March 2018

---- CORRESPONDENT Michael Pearce
Example 4: every day for hours and hours, Sarah tried her hardest to
"""


DSCORER_RESPONSE = """The following examples contain male pronouns and names:

Example 3, Example 4.~

FINAL ANSWER: [3, 4]
"""


def get_detection_template(examples, explanation):
  template =  f"""
  <|begin_of_text|>

  <|start_header_id|>system<|end_header_id|>
  \n\n{DSCORER_SYSTEM_PROMPT}
  <|eot_id|>

  <|start_header_id|>user<|end_header_id|>
  \n\n{DSCORER_EXAMPLE}
  <|eot_id|>

  <|start_header_id|>assistant<|end_header_id|>
  \n\n{DSCORER_RESPONSE}"
  <|eot_id|>

  <|start_header_id|>user<|end_header_id|>
  \n\nFeature explanation: {explanation}\n\nText examples:\n\n{examples}
  <|eot_id|>

  <|start_header_id|>assistant<|end_header_id|>
  \n\n"""

  return template