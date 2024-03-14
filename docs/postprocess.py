import re
from pathlib import Path
from typing import Match, Union

from bs4 import BeautifulSoup, Tag

# Install from https://github.com/carpedm20/emoji/
# with pip install emoji
try:
    from emoji import emojize
except ImportError:
    print("Error: package not found, install 'emoji' package with 'pip install emoji'")


def match_to_emoji(m: Match[str]) -> str:
    """Call emoji.emojize on m)."""
    return emojize(m.group())


def emojize_all(s: str) -> str:
    """Convert all emojis :aliases: of the string s to emojis in UTF-8."""
    return re.sub(r":([a-z_-]+):", match_to_emoji, s)


def update_image_paths(soup: BeautifulSoup) -> None:
    """Update src attribute of img tags starting with 'utils/'."""
    imgs = soup.find_all("img", src=re.compile("^utils/"))
    for img in imgs:
        if isinstance(img, Tag):  # Ensuring that 'img' is a Tag object
            img["src"] = img["src"].replace("utils/", "_images/")


def process_html_file(html_file: Union[str, Path]) -> None:
    """Process an individual HTML file."""
    with open(html_file, "r", encoding="utf-8") as file:
        content = file.read()

    # Convert emojis in the entire HTML content
    content = emojize_all(content)

    soup = BeautifulSoup(content, "html.parser")

    # Update all <img> tags with src starting with "utils/"
    update_image_paths(soup)

    # Write the changes back to the HTML file
    with open(html_file, "w", encoding="utf-8") as file:
        file.write(str(soup))


if __name__ == "__main__":
    # Specify the pattern to match the HTML files you want to postprocess
    __location__: Path = Path(__file__).parent
    html_files: list[Path] = list((__location__ / "_build" / "html").glob("*.html"))

    for html_file in html_files:
        process_html_file(html_file)

    print("HTML postprocessing completed.")
