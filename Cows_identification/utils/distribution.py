import os
import json
import csv

root_folder = '/home/mine01/Desktop/code/AWP/Cows_identification/data/limit_2videos/test'  # update this path
result_file_path = os.path.join(root_folder, 'result_30frames.csv')

# Create the CSV file and write the header
with open(result_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Type", "Probability"])

    # Iterate over subfolders in the root folder
    for folder_name in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, folder_name)

        # Proceed only if it's a folder
        if os.path.isdir(subfolder_path):
            # Iterate over files in the subfolder
            for file_name in os.listdir(subfolder_path):
                # Proceed only if it's a .json file
                if file_name.endswith('.json'):
                    file_path = os.path.join(subfolder_path, file_name)

                    # Open and load the JSON file
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)

                        # Get first key-value pair from the JSON object
                        first_key, first_value = next(iter(data.items()))

                        # Check if the key matches the folder name
                        # and write the result to the CSV file
                        if folder_name == first_key:
                            writer.writerow(["R", first_value])
                        else:
                            writer.writerow(["W", first_value])
