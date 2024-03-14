import os
import re

from bs4 import BeautifulSoup

# Install from https://github.com/carpedm20/emoji/
# with pip install emoji
try:
    from emoji import emojize
except ImportError:
    print("Error: package not found, install 'emoji' package with 'pip install emoji'")


def match_to_emoji(m):
    """Call emoji.emojize on m)."""
    return emojize(m.group())


def emojize_all(s):
    """Convert all emojis :aliases: of the string s to emojis in UTF-8."""
    return re.sub(r":([a-z_-]+):", match_to_emoji, s)


# Specify the pattern to match the HTML files you want to postprocess
__location__ = os.path.dirname(__file__)

html_files = [os.path.join(__location__, "_build", "html", "readme.html")]

for html_file in html_files:
    with open(html_file, "r", encoding="utf-8") as file:
        content = file.read()

    # Convert emojis in the entire HTML content
    content = emojize_all(content)

    soup = BeautifulSoup(content, "html.parser")

    # Find all <img> tags with the specific src attribute
    imgs = soup.find_all("img", {"src": "utils/equiadapt_logo.png"})

    for img in imgs:
        # Update the src attribute to the correct path
        img["src"] = "_images/equiadapt_logo.png"

    # Write the changes back to the HTML file
    with open(html_file, "w", encoding="utf-8") as file:
        file.write(str(soup))

print("HTML postprocessing completed.")
