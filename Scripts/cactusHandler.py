import re
from datetime import datetime

def parse_cactus_file(file_path):
    cme_events = []
    with open(file_path, 'r') as f:
        for line in f:
            # Match lines starting with CME ID (4-digit number)
            if re.match(r"\s*\d{4}\|", line):
                parts = line.strip().split('|')
                if len(parts) < 10:
                    continue  # Ensure enough columns
                halo = parts[-1].strip()  
                if halo:
                    try:
                        t0 = parts[1].strip()
                        t0_dt = datetime.strptime(t0, "%Y/%m/%d %H:%M")
                        t0_dt = t0_dt.replace(tzinfo=None)
                        cme_events.append(t0_dt)
                    except Exception as e:
                        print("Error parsing line:", line)
    return cme_events
