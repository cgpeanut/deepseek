import json

# Open and read the JSON file
with open('powerball_numbers.json', 'r') as file:
    data = json.load(file)

# Print the data
print(data)
