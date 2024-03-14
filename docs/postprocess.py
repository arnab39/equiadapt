import os

from bs4 import BeautifulSoup

# Specify the pattern to match the HTML files you want to postprocess
__location__ = os.path.dirname(__file__)

html_files = [os.path.join(__location__, "_build", "html", "readme.html")]

for html_file in html_files:
    with open(html_file, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Find all <img> tags with the specific src attribute
    imgs = soup.find_all("img", {"src": "utils/equiadapt_logo.png"})

    for img in imgs:
        # Update the src attribute to the correct path
        img["src"] = "_images/equiadapt_logo.png"

    # Write the changes back to the HTML file
    with open(html_file, "w", encoding="utf-8") as file:
        file.write(str(soup))

print("HTML postprocessing completed.")
