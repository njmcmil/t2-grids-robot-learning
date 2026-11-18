import sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Preserve the working native-library import order.
import torch,torchvision
from t2.train import train
from t2.models import VARIANTS
parser=argparse.ArgumentParser()
parser.add_argument('--variant',choices=VARIANTS,required=True)
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--smoke',action='store_true')
args=parser.parse_args()
print(train(ROOT,args.variant,args.seed,args.epochs,smoke=args.smoke))
