"""
Script that processes the Project Gutenberg files into fewer larger files.
strips Gutenberg boilerplate text from each book.
"""

import argparse
import os
import re
from tqdm import tqdm

# Try to import gutenberg library, but provide fallback
try:
    from gutenberg.src.cleanup import strip_headers # type: ignore
    GUTENBERG_AVAILABLE = True
except ImportError:
    GUTENBERG_AVAILABLE = False
    print("Warning: gutenberg library not available. Using fallback boilerplate stripping.")


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Strip Project Gutenberg boilerplate text from beginning and end of book.
    Fallback implementation if gutenberg library is not available.
    """
    # Common start markers
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    
    # Common end markers
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    
    # Find start position
    start_pos = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            # Find end of line after marker
            newline_pos = text.find('\n', pos)
            if newline_pos != -1:
                start_pos = newline_pos + 1
                break
    
    # Find end position
    end_pos = len(text)
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            end_pos = pos
            break
    
    return text[start_pos:end_pos].strip()


def is_english(text, threshold=0.9):
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) > threshold


def combine_files(file_paths, target_dir, max_size_mb=500, separator="<|endoftext|>", 
                  fallback_encoding="latin1", strip_boilerplate=True):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    current_content = []
    current_size = 0
    file_counter = 1

    for file_path in tqdm(file_paths):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            # Attempt to read the file with a fallback encoding
            tqdm.write(f"Warning: UnicodeDecodeError encountered. Trying fallback encoding for {file_path}")
            with open(file_path, "r", encoding=fallback_encoding) as file:
                content = file.read()

        if not is_english(content):
            tqdm.write(f"Skipping {file_path} as it does not contain primarily English text.")
            continue
        
        # Strip Gutenberg boilerplate
        if strip_boilerplate:
            if GUTENBERG_AVAILABLE:
                content = strip_headers(content)
            else:
                content = strip_gutenberg_boilerplate(content)

        # Regular expression to replace multiple blank lines with a single blank line
        content = re.sub(r"\n\s*\n", "\n\n", content)
        estimated_size = len(content.encode("utf-8"))

        if current_size + estimated_size > max_size_mb * 1024 * 1024:
            target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
            with open(target_file_path, "w", encoding="utf-8") as target_file:
                target_file.write(separator.join(current_content))
            file_counter += 1
            current_content = [content]
            current_size = estimated_size
        else:
            current_content.append(content)
            current_size += estimated_size

    if current_content:
        target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
        with open(target_file_path, "w", encoding="utf-8") as target_file:
            target_file.write(separator.join(current_content))
    return file_counter


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess and combine text files for pretraining")

    parser.add_argument("--data_dir", type=str, default="gutenberg/data/raw",
                        help="Directory containing the downloaded raw training data")
    parser.add_argument("--max_size_mb", type=int, default=500,
                        help="The maximum file size for each concatenated file in megabytes")
    parser.add_argument("--output_dir", type=str, default="gutenberg_preprocessed",
                        help="Directory where the preprocessed data will be saved")
    parser.add_argument("--strip_boilerplate", action="store_true", default=True,
                        help="Strip Project Gutenberg boilerplate text from books")

    args = parser.parse_args()

    all_files = [os.path.join(path, name) for path, subdirs, files in os.walk(args.data_dir)
                 for name in files if name.endswith((".txt", ".txt.utf8"))]

    print(f"{len(all_files)} file(s) to process.")
    file_counter = combine_files(
        all_files, 
        args.output_dir, 
        max_size_mb=args.max_size_mb,
        strip_boilerplate=args.strip_boilerplate
    )
    print(f"{file_counter} file(s) saved in {os.path.abspath(args.output_dir)}")