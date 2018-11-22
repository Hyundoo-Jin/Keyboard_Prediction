import os, sys

def argument_setting(arguments) :
    if arguments.language == 'kor' or 'k' :
        language = 'kor'
    elif arguments.language == 'eng' or 'e' :
        language = 'eng'
    else :
        raise 