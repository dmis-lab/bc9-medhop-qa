import csv

class MedHopDataset:
    def __init__(self):
        self.data = []

    def load_csv(self, path):
        self.data = []

        with open(path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                self.data.append({
                    "qidx": int(row["QIDX"]),
                    "question": row["Question"].strip(),
                })
