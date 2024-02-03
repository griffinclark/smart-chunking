from llama_hub.pdf_table import PDFTableReader
from pathlib import Path
import os

reader = PDFTableReader()
pdf_path = os.path.join(os.path.dirname(__file__), '../uber_data/uber_2021.pdf')
documents = reader.load_data(file=pdf_path)

output_file = "output.txt"
with open(output_file, "w") as file:
    file.write(str(documents))
