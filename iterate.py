# TODO: Make the path interchangable
import pathlib
import os

currentdirectory = pathlib.Path("/ImageSamples/")

nr = 0
for currentfile in currentdirectory.iterdir():
	os.system("python3 run.py --image " + currentfile + "--output_json --input_nr" + nr)
	nr += 1


