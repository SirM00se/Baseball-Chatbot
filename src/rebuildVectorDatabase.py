import sys
import subprocess

python_executable = sys.executable  # current Python interpreter
subprocess.run([python_executable, "dataScraper.py"])
subprocess.run([python_executable, "vectordatabase.py"])
subprocess.run([python_executable, "SQLConversion.py"])
