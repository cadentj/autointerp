# %%

import os

def get_html_files(base_dir):
    html_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.html'):
                html_files.append(os.path.join(root, file))
    return html_files

# Usage
base_directory = './working/results'  # Replace with your base directory if not the current directory
html_file_paths = get_html_files(base_directory)

# %%

import shutil
def create_index_html(paths, output_dir):

    links = ""
    for path in paths:

        feature = path.split('/')[-2]

        shutil.copy(path, f'./docs/{feature}.html')

        new_path = f'{feature}.html'

        links += f'<li><a href="{new_path}">{feature}</a></li>'

    html_content = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Results Index</title>
    </head>
    <body>
        <h1>Results Index</h1>
        <ul>
            {''.join(links)}
        </ul>
    </body>
    </html>
    '''

    with open(output_dir, 'w') as f:
        f.write(html_content)

create_index_html(html_file_paths, './docs/index.html')