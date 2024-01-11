import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import Defaults, GPU

Defaults(name="2024v32_mar3", GPU=GPU.v32, epochs = 1000, lr=0.003)
Defaults(name="2024a80_mar3", GPU=GPU.a80, epochs = 1000, lr=0.003)
Defaults(name="2024v16_mar3", GPU=GPU.v16, epochs = 1000, lr=0.003)
Defaults(name="2024a40_mar3", GPU=GPU.a40, epochs = 1000, lr=0.003)

