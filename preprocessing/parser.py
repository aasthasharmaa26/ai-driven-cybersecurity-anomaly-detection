import re
import re

def clean_log(log):

    log = log.lower()

    # remove timestamps
    log = re.sub(r'\[.*?\]', '', log)

    # replace IPs
    log = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', log)

    # replace numbers
    log = re.sub(r'\d+', '<NUM>', log)

    # remove extra spaces
    log = re.sub(r'\s+', ' ', log).strip()

    return log

def extract_template(log):

    patterns = {

        "failed login":
        r'failed.*password',

        "brute force":
        r'brute force',

        "privilege escalation":
        r'privilege escalation',

        "port scan":
        r'port scan',

        "unauthorized access":
        r'unauthorized',

    }

    for template, pattern in patterns.items():

        if re.search(pattern, log):

            return template

    return "normal activity"