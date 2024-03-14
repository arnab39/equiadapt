import argparse  # Import the argparse module
import re
from pathlib import Path
from typing import Match, Union

from bs4 import BeautifulSoup, Tag

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

    content = emojize_all(content)
    soup = BeautifulSoup(content, "html.parser")
    update_image_paths(soup)

    with open(html_file, "w", encoding="utf-8") as file:
        file.write(str(soup))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HTML files.")
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the directory containing HTML files to process.",
    )

    args = parser.parse_args()

    if args.path:
        base_path = Path(args.path)
    else:
        __location__: Path = Path(__file__).parent
        base_path = __location__ / "_build" / "html"

    html_files: list[Path] = list(base_path.glob("*.html"))

    for html_file in html_files:
        print(f"Processing {html_file}...")
        process_html_file(html_file)

    print("HTML postprocessing completed.")
