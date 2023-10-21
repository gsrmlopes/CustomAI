import os
import argparse


def create_structure(input_text, output_path):
    lines = input_text.split('\n')
    current_path = output_path
    for line in lines:
        # Count the number of indents
        indents = line.count('|   ')
        name = line.split('-- ')[-1]
        # If it's a directory
        if name.endswith('/'):
            # Remove the trailing slash
            name = name[:-1]
            # Update the current path
            current_path = '/'.join(current_path.split('/')[:indents + 1] + [name])
            # Create the directory if it doesn't exist
            if not os.path.exists(current_path):
                os.makedirs(current_path)
        else:
            # Update the current path
            file_path = '/'.join(current_path.split('/')[:indents + 1] + [name])
            # Create the file if it doesn't exist
            if not os.path.exists(file_path):
                with open(file_path, 'a', encoding='utf-8'):
                    pass


def main():
    parser = argparse.ArgumentParser(description='Create directory structure.')
    parser.add_argument('--Read', type=str, required=True, help='Path to the input text file.')
    parser.add_argument('--Output', type=str, required=True, help='Path to the output directory.')
    args = parser.parse_args()

    with open(args.Read, 'r') as f:
        input_text = f.read()

    create_structure(input_text, args.Output)


if __name__ == "__main__":
    main()
