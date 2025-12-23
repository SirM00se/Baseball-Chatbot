import re
import csv
ensp = "\u2002"
pattern = re.compile(r"^\d+\.\d+" + ensp)
appendline = ""
chunk = ""
tempchunk = ""
try:
    writeFile = open("../data/baseballrules.csv", 'a', newline="", encoding="utf-8")
    writer = csv.writer(writeFile)
    with open("../data/BaseballRules.txt", encoding="utf-8") as f:
        for line in f:
            if pattern.search(line):
                chunk = tempchunk
                appendline = f"https://img.mlbstatic.com/mlb-images/image/upload/mlb/atcjzj9j7wrgvsm8wnjq.pdf",chunk, "rule"
                if chunk != "":
                    writer.writerow(appendline)
                tempchunk = ""
                tempchunk += " "+line.strip()
            else:
                tempchunk += " "+line.strip()
finally:
    chunk = tempchunk
    appendline = f"https://img.mlbstatic.com/mlb-images/image/upload/mlb/atcjzj9j7wrgvsm8wnjq.pdf",chunk, "rule"
    if chunk != "":
        writer.writerow(appendline)
    chunk = ""
    tempchunk = ""
    writeFile.close()
try:
    writeFile = open("../data/baseballrules.csv", 'a', newline="", encoding="utf-8")
    writer = csv.writer(writeFile)
    with open("../data/Definition of Terms.txt", encoding="utf-8") as f:
        for line in f:
            if ensp in line:
                chunk = tempchunk
                appendline = f"https://img.mlbstatic.com/mlb-images/image/upload/mlb/atcjzj9j7wrgvsm8wnjq.pdf",chunk, "rule"
                if chunk != "":
                    writer.writerow(appendline)
                tempchunk = ""
                tempchunk += " "+line.strip()
            else:
                tempchunk += " "+line.strip()
finally:
    chunk = tempchunk
    appendline = f"https://img.mlbstatic.com/mlb-images/image/upload/mlb/atcjzj9j7wrgvsm8wnjq.pdf",chunk, "rule"
    if chunk != "":
        writer.writerow(appendline)
    chunk = ""
    tempchunk = ""
    writeFile.close()