from preprocessing.parser import clean_log
from preprocessing.parser import extract_template

with open("logs/live.log", "r") as f:

    lines = f.readlines()

    for line in lines[-10:]:

        cleaned = clean_log(line)

        template = extract_template(cleaned)

        print("LOG:", cleaned)

        print("TYPE:", template)

        print("-" * 50)