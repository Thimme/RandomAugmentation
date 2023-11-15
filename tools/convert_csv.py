import json
import csv
import argparse

# Function to convert a JSON file to CSV
def convert_json_to_csv(input_file, output_file):
    with open(input_file, 'r') as json_file:
        data = [json.loads(line) for line in json_file]

    if not data:
        print("No data found in the input file.")
        return

    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())

        # Write the header
        writer.writeheader()

        # Write the data
        for row in data:
            writer.writerow(row)

    print(f"CSV file '{output_file}' has been created successfully.")


if __name__ == "__main__":
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(description="Convert a JSON file to a CSV file.")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Path to the output CSV file")

    args = parser.parse_args()

    # Call the function to convert JSON to CSV
    convert_json_to_csv(args.input_file, args.output_file)
