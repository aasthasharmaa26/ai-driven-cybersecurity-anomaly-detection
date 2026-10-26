with open("logs/live.log", "r") as f:

    lines = f.readlines()

    for line in lines[-5:]:

        print(line)