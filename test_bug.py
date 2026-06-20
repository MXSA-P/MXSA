import re
from simba.voice.command_parser import CommandParser
parser = CommandParser()
print(parser.parse("hello-simba"))
print(parser.parse("clear-path"))
print(parser.parse("go-forward"))
print(parser.parse("let's-play"))
print(parser.parse("high-speed"))
print(parser.parse("speed-up"))
