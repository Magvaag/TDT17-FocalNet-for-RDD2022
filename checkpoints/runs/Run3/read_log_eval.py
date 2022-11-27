import json

filename = "Run.log.json"

with open(filename) as f:
    # Load the lines of the file into a list.
    lines = f.readlines()

    # Split the lines into a list of dictionaries.
    data = [json.loads(line) for line in lines]

    for i in range(len(data)):
        if data[i]["mode"] == "train":
            # print(data[i]["loss_bbox"])
            print(data[i]["epoch"]-1 + data[i]["iter"] / 2300.0)
