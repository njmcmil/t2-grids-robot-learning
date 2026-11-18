import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch,torchvision
from t2.evaluate import evaluate_selected
from t2.results import summarize
evaluate_selected(ROOT)
summarize(ROOT)
