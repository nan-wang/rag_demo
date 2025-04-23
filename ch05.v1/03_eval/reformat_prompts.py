import json

def load_and_reformat_json(input_filepath, output_filepath):
    """
    Loads a JSON file, reformats it with UTF-8 encoding, and writes it to a new file.

    Parameters:
    - input_filepath: str, path to the input JSON file.
    - output_filepath: str, path where the reformatted JSON will be saved.
    """
    try:
        # Load the JSON data from the input file with UTF-8 encoding
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        print(f"Successfully loaded JSON data from {input_filepath}.")

        # remove the markdown marks
        data["examples"]

        # Write the JSON data to the output file with UTF-8 encoding
        # ensure_ascii=False preserves Unicode characters
        # indent=4 makes the JSON file human-readable
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        print(f"Reformatted JSON data has been written to {output_filepath}.")

    except FileNotFoundError as e:
        print(f"Error: The file {input_filepath} does not exist. {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


from pathlib import Path
import glob

# Example usage
if __name__ == "__main__":
    for doc in glob.glob("/Users/nanwang/.cache/ragas/chinese/*.json"):
        input_file = doc
        output_file = f'./chinese/{Path(input_file).name}'
        load_and_reformat_json(input_file, output_file)
    # input_file = "/Users/nanwang/.cache/ragas/chinese/keyphrase_extraction.json"

