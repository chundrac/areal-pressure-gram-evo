#!usr/bin/env python
# -*- coding: utf-8 -*-

import sys

def clean(text):
    text = text.replace('=','$=$')
    text = text.replace('_','$\\underline{\\phantom{X}}$')
    text = text.replace('≠','$\\neq$')
    text = text.replace('<','$>$')
    return text

def main():
    f = open(sys.argv[1],'r')
    text = f.read()
    f.close()
    cleaned_text = clean(text)
    f = open(sys.argv[1],'w')
    f.write(cleaned_text)
    f.close()

if __name__ == "__main__":
    main()
