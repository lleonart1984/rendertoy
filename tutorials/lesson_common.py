# If file is run as a script add parent directory to path
# This allow import rendering module
import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
ROOT_DIR = str(parentdir)
