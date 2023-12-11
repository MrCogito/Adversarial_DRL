import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import Defaults, GPU

Defaults(name="Test1", GPU=GPU.a80)
