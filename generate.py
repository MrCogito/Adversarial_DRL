import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import Defaults, GPU

Defaults(name="1Test2024v32", GPU=GPU.v32, random=False)
Defaults(name="1Test2024v80", GPU=GPU.a80, random=False)

