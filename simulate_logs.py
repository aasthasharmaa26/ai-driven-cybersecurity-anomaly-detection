import time
import random
from datetime import datetime

logs = [

    "Login successful for admin",

    "Failed password for root from 192.168.1.5",

    "Multiple failed login attempts detected",

    "Port scan activity detected",

    "Privilege escalation attempt",

    "Unauthorized access attempt",

    "Suspicious outbound traffic spike",

    "Possible brute force attack detected",

    "Repeated SSH authentication failures",

    "Unusual midnight access pattern",

    "Admin access from unknown IP",

    "High-frequency requests from single IP",

    "Abnormal database query activity",

]

while True:

    log = random.choice(logs)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    final_log = f"[{timestamp}] {log}"

    with open("logs/live.log", "a") as f:

        f.write(final_log + "\n")

    print(final_log)

    time.sleep(2)