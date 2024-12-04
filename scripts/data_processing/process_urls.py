# -*- coding: utf-8 -*-
# @Time    : 2024/8/24 8:36
# @Project : AnimateMaster
# @FileName: process_urls.py

"""
python scripts/data_processing/process_urls.py \
data/youtube_0901/raw_url.txt
"""

import os
import argparse
import subprocess


def process_url(url, output_dir):
    # Extract the name from the URL for the output file
    url_name = url.split('@')[-1].split('/')[-2]
    output_file = os.path.join(output_dir, f"{url_name}.txt")
    if os.path.exists(output_file):
        return
    # Run the yt-dlp command
    command = [
        'yt-dlp', '-s', '--flat-playlist', '--print-to-file', 'url',
        output_file, url
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"Processed: {url} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to process {url}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Process URLs from a txt file using yt-dlp.')
    parser.add_argument('input_file', type=str, help='Path to the input txt file containing URLs.')
    parser.add_argument('--output_dir', type=str, help='Directory where output files will be saved.')

    args = parser.parse_args()

    # Ensure the output directory exists
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input_file), "urls")
    os.makedirs(args.output_dir, exist_ok=True)

    # Read and process each URL from the input file
    with open(args.input_file, 'r') as file:
        urls = file.readlines()
        for url in urls:
            url = url.strip()
            if url:
                process_url(url, args.output_dir)


if __name__ == "__main__":
    main()
