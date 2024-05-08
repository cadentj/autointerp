# Define the data as a list of dictionaries
neuron_data = [
    {
        'layer': 8,
        'index': 15269,
        'max_activation': '6:f02',
        'explanations': [
            {
                'score': (0.8, 0.2),
                'generation_score': 1.42798125743686597,
                'text': 'Punctuation marks or abbreviations that separate or precede numerical values or website addresses in references to publications or websites.'
            }
        ]
    },
    # Add more neuron data as needed
]

# Generate HTML content
html_content = '<html><head><title>Neuron Feature Visualization</title></head><body>'
html_content += '<h1>Neuron Feature Visualization</h1>'

for neuron in neuron_data:
    html_content += f"<h2>Layer {neuron['layer']}, Index {neuron['index']}</h2>"
    html_content += f"<p><strong>Max Activation:</strong> {neuron['max_activation']}</p>"
    html_content += '<ul>'
    for exp in neuron['explanations']:
        html_content += '<li>'
        html_content += f"Detection Score: {exp['score']}, Generation Score: {exp['generation_score']}<br>"
        html_content += f"Explanation: {exp['text']}"
        html_content += '</li>'
    html_content += '</ul>'

html_content += '</body></html>'

# Save the HTML to a file
with open('neuron_visualization.html', 'w') as file:
    file.write(html_content)

print("HTML file 'neuron_visualization.html' has been created.")
