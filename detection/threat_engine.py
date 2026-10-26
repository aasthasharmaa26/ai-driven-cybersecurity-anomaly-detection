def classify_threat(log):

    log = log.lower()

    if "brute" in log:

        return {

            "threat_type": "Brute Force",

            "severity": "High",

            "risk_score": 90,

            "mitre_attack": "T1110"

        }

    elif "root" in log:

        return {

            "threat_type": "Privilege Escalation",

            "severity": "Critical",

            "risk_score": 95,

            "mitre_attack": "T1068"

        }

    elif "admin" in log:

        return {

            "threat_type": "Unauthorized Admin Access",

            "severity": "High",

            "risk_score": 88,

            "mitre_attack": "T1078"

        }

    else:

        return {

            "threat_type": "Unknown Threat",

            "severity": "Low",

            "risk_score": 30,

            "mitre_attack": "N/A"

        }