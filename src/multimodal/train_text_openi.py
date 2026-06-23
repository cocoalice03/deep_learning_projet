import argparse, os, sys
import numpy as np, pandas as pd, torch, mlflow
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "data/huggingface_cache"))

from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(PROJECT_ROOT)

from src.utils.common import LABELS, configure_mlflow, seed_everything, get_device, ensure_dir
from src.evaluation.metrics import multilabel_metrics

def finite_log_metric(name, value, step=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if np.isfinite(value):
        mlflow.log_metric(name, value, step=step)

class TextDS(Dataset):
    def __init__(self, df, tok, max_len=256):
        self.df=df.reset_index(drop=True); self.tok=tok; self.max_len=max_len
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        enc=self.tok(str(r.text), truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        y=torch.tensor(r[LABELS].values.astype("float32"))
        return {k:v.squeeze(0) for k,v in enc.items()}, y

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv", default="data/openi_metadata.csv"); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch_size", type=int, default=8); args=ap.parse_args()
    seed_everything(42); ensure_dir("checkpoints")
    df=pd.read_csv(args.csv); tr, va=train_test_split(df, test_size=.2, random_state=42)
    tok=AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model=AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(LABELS), problem_type="multi_label_classification")
    device=get_device(); model.to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=2e-5)
    trl=DataLoader(TextDS(tr,tok), batch_size=args.batch_size, shuffle=True); val=DataLoader(TextDS(va,tok), batch_size=args.batch_size)
    configure_mlflow()
    mlflow.set_experiment("openi_multimodal")
    with mlflow.start_run(run_name="text_only_distilbert"):
        mlflow.log_params(vars(args))
        for e in range(args.epochs):
            model.train()
            for enc,y in trl:
                enc={k:v.to(device) for k,v in enc.items()}; y=y.to(device)
                out=model(**enc, labels=y); opt.zero_grad(); out.loss.backward(); opt.step()
            model.eval(); probs=[]; ys=[]
            with torch.no_grad():
                for enc,y in val:
                    enc={k:v.to(device) for k,v in enc.items()}
                    logits=model(**enc).logits
                    probs.append(torch.sigmoid(logits).cpu().numpy()); ys.append(y.numpy())
            m=multilabel_metrics(np.vstack(ys), np.vstack(probs))
            for k,v in m.items(): finite_log_metric("val_"+k,v,e)
            print(e+1,m)
        model.save_pretrained("checkpoints/text_only_distilbert"); tok.save_pretrained("checkpoints/text_only_distilbert")
        mlflow.log_artifacts("checkpoints/text_only_distilbert")
if __name__=="__main__": main()
