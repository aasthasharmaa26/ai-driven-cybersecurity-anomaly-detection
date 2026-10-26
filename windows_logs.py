import requests
import win32evtlog

server = 'localhost'

# USE SYSTEM INSTEAD OF SECURITY
log_type = 'System'

hand = win32evtlog.OpenEventLog(server, log_type)

flags = win32evtlog.EVENTLOG_BACKWARDS_READ | \
        win32evtlog.EVENTLOG_SEQUENTIAL_READ

events = win32evtlog.ReadEventLog(hand, flags, 0)

count = 0

for event in events:

    try:

        event_id = event.EventID & 0xFFFF

        source = event.SourceName

        time_generated = str(event.TimeGenerated)

        log_message = f"""

        Event ID: {event_id}

        Source: {source}

        Time: {time_generated}

        """

        print(log_message)

        response = requests.post(

            "http://127.0.0.1:8000/detect",

            params={

                "log": log_message

            }

        )

        print(response.json())

        print("-" * 50)

        count += 1

        if count >= 10:

            break

    except Exception as e:

        print("ERROR:", e)