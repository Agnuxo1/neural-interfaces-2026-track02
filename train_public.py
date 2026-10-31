"""Train, verify and package a genuine PROTEUS BCI classifier using public data.

Training and model selection use ONLY the organisers' train/validation
splits, with zero access to the test labels until hyperparameters are frozen.
All intermediate features, split hashes, metrics and trained coefficients
are retained for reproducibility. No Server-IA or cloud accelerator.
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
import warnings
import zipfile
from pathlib import Path

os.environ["NEURALBENCH_EXTRACT_CPUS"]="1"
os.environ["OMP_NUM_THREADS"]="6"
os.environ["MKL_NUM_THREADS"]="6"
os.environ["OPENBLAS_NUM_THREADS"]="6"
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

WORK=Path(__file__).resolve().parent
OUT=WORK/"training_results.json"
KIT=Path(os.environ.get("BCI_KIT_DIR", str(WORK/"2026-competition"))).resolve()
SAMPLES=Path(os.environ.get("BCI_DATA_DIR", str(WORK/"data"))).resolve()
SEED=int(os.environ.get("BCI_SEED","20261031"))
if not (KIT/"tracks"/"bci_decoding"/"benchmark_utils"/"nb_task.py").exists():
    raise RuntimeError("Set BCI_KIT_DIR to the original competition kit clone")
if not SAMPLES.is_dir():
    raise RuntimeError("Set BCI_DATA_DIR to the downloaded PROTEUS public dataset")
sys.path.insert(0,str(KIT/"tracks"/"bci_decoding"))
sys.path.insert(0,str(WORK))
import kit_compat
kit_compat.apply()
from benchmark_utils.nb_task import load_task

spec=importlib.util.spec_from_file_location("my_bci_model",WORK/"submission.py")
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)

CONTEXTS={"AcquisitionBH":"BrainHero", "OnlineRawBH":"BrainHero",
          "AcquisitionGraz":"Graz", "OnlineRawGraz":"Graz"}


def context_from_trigger(trigger):
    """Map BIDS protocol event to its official 2-interface cell identifier."""
    return CONTEXTS[trigger["task"]]


def run_from_trigger(trigger):
    """Maintain official acquisition order when the test loader is streamed."""
    return int(trigger["run"])


def sample_features(loader, rate, split):
    """Extract only public split features without retaining raw EEG arrays."""
    xs,ys,subjects,streams,contexts=[],[],[],[],[]
    start=time.time()
    for step,(batch,y,info) in enumerate(loader):
        x=module.extract_bci_features(batch.detach().cpu().numpy(),rate)
        xs.append(x)
        ys.append(y.detach().cpu().numpy().reshape(-1).astype(np.int64))
        subjects.append(info["subject_id"].detach().cpu().numpy().reshape(-1))
        streams.append(info["stream_id"].detach().cpu().numpy().reshape(-1))
        contexts.append(info["context_id"].detach().cpu().numpy().reshape(-1))
        if step%75==0:
            print(split,"batch",step,"examples",sum(len(v) for v in ys),
                  "elapsed",round(time.time()-start,1),flush=True)
    rec={
      "X":np.concatenate(xs,axis=0),"y":np.concatenate(ys,axis=0),
      "subject":np.concatenate(subjects,axis=0),
      "stream":np.concatenate(streams,axis=0),
      "context":np.concatenate(contexts,axis=0)
    }
    print(split,"total",len(rec["y"]),"classes",np.unique(rec["y"],return_counts=True),
          "subjects",np.unique(rec["subject"]),"features",rec["X"].shape,flush=True)
    np.savez_compressed(WORK/(split+"_features.npz"),**rec)
    return rec


def score(y,p,sub,stream,ctx):
    """Recalculate independently the official subject×session×context mean."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        groups=np.stack((sub,stream,ctx),axis=1)
        cells=np.unique(groups,axis=0)
        scores=[]
        for g in cells:
            mask=(groups==g).all(axis=1)
            if not mask.any():
                continue
            scores.append(float(balanced_accuracy_score(y[mask],p[mask])))
    return {"bacc_cell_mean":float(np.mean(scores)),"n_cells":len(scores),
            "bacc_pooled":float(balanced_accuracy_score(y,p)),
            "n_examples":len(y)}


def model_train(train,val,rate,n_classes):
    """Choose regularisation from validation ONLY and freeze before test."""
    scaler=StandardScaler().fit(train["X"])
    X=scaler.transform(train["X"]).astype(np.float64)
    V=scaler.transform(val["X"]).astype(np.float64)
    candidates=[]
    for c in (0.005,0.05,0.5,5.):
        reg=LogisticRegression(C=c,max_iter=1200,class_weight="balanced",
                               random_state=SEED,solver="lbfgs")
        reg.fit(X,train["y"])
        pv=reg.predict(V)
        metrics=score(val["y"],pv,val["subject"],val["stream"],val["context"])
        candidates.append({"C":c,**metrics})
        print("VAL",c,metrics,flush=True)
    best=max(candidates,key=lambda r:(r["bacc_cell_mean"],-r["C"]))
    # Freeze the chosen C; then train on BOTH permitted train+validation labels.
    # Neither hyperparameter selection nor training reads the official test.
    merged_X=np.concatenate((train["X"],val["X"]),axis=0)
    merged_y=np.concatenate((train["y"],val["y"]),axis=0)
    final_scaler=StandardScaler().fit(merged_X)
    final_model=LogisticRegression(
        C=best["C"],max_iter=1200,class_weight="balanced",
        random_state=SEED,solver="lbfgs"
    ).fit(final_scaler.transform(merged_X).astype(np.float64),merged_y)
    if len(final_model.classes_)!=n_classes:
        raise RuntimeError("Missing one or more BCI classes from public training")
    np.savez_compressed(
        WORK/"weights.npz",
        center=final_scaler.mean_.astype(np.float64),
        scale=final_scaler.scale_.astype(np.float64),
        coef=final_model.coef_.astype(np.float64),
        intercept=final_model.intercept_.astype(np.float64),
        classes=final_model.classes_.astype(np.int64)
    )
    return candidates,best


def main():
    start=time.time()
    print("Official NeuralBench data source",SAMPLES,flush=True)
    loaders,meta=load_task(
        "eeg","_motor_imagery_stream",data_dir=SAMPLES,device="cpu",
        batch_size=64,num_workers=0,seed=SEED,subset="full",
        target_transform=lambda y:y.argmax(-1),
        context_of=context_from_trigger,order_of=run_from_trigger
    )
    rate=float(meta["sfreq"])
    classes=int(meta["raw_target_shape"][-1])
    print("META rate",rate,"chans",meta["n_chans"],"samples",meta["n_times"],
          "classes",classes,flush=True)
    assert meta["n_chans"]==41 and meta["n_times"]==480
    tr=sample_features(loaders["train"],rate,"train")
    val=sample_features(loaders["val"],rate,"val")
    shared=np.intersect1d(tr["subject"],val["subject"])
    if shared.size:
        raise RuntimeError(f"Train/validation subject leakage: {shared}")
    candidates,best=model_train(tr,val,rate,classes)
    # Load frozen weights with the exact Zip submission model and evaluate once.
    with np.load(WORK/"weights.npz",allow_pickle=False) as archive:
        frozen={k:archive[k] for k in archive.files}
    estimator=module.SpectralBCIModel(frozen,meta["n_chans"],meta["n_times"],rate)
    te=sample_features(loaders["test"],rate,"test")
    if (np.intersect1d(tr["subject"],te["subject"]).size or
            np.intersect1d(val["subject"],te["subject"]).size):
        raise RuntimeError("Subject leakage between training and held-out test")
    tepred=frozen["classes"][np.argmax(
        (te["X"].astype(float)-frozen["center"])/np.maximum(frozen["scale"],1e-5)
        @frozen["coef"].T + frozen["intercept"],axis=1)]
    test_score=score(te["y"],tepred,te["subject"],te["stream"],te["context"])
    print("TEST (once)",test_score,flush=True)
    record={
        "seed":SEED,"study":"Dreyer2026Proteus/NEMAR nm000290 v1.0.0",
        "protocol":"official neuralbench _motor_imagery_stream; cross-subject split",
        "train_subjects":tr["subject"].astype(int).tolist()[:0]+np.unique(tr["subject"]).astype(int).tolist(),
        "val_subjects":np.unique(val["subject"]).astype(int).tolist(),
        "test_subjects":np.unique(te["subject"]).astype(int).tolist(),
        "train_rows":len(tr["y"]),"validation_rows":len(val["y"]),
        "test_rows":len(te["y"]),"features":int(tr["X"].shape[1]),
        "validation_candidates":candidates,"selected_validation":best,
        "heldout_test":test_score,"test_confusion_matrix":confusion_matrix(te["y"],tepred).tolist(),
        "elapsed_seconds":round(time.time()-start,2),
        "source":"D:/PROJECTS/.cognition/concursos-24h/chatgpt/track02_model/submission.py",
    }
    OUT.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    zipname=WORK/"Agnuxo_SpectralBCI_Track02.zip"
    with zipfile.ZipFile(zipname,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.write(WORK/"submission.py","submission.py")
        z.write(WORK/"weights.npz","weights.npz")
    with zipfile.ZipFile(zipname) as z:
        assert z.testzip() is None and sorted(z.namelist())==["submission.py","weights.npz"]
    digest=hashlib.sha256(zipname.read_bytes()).hexdigest()
    record["submission_zip"]=str(zipname)
    record["submission_zip_sha256"]=digest
    OUT.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print("PACKAGE_COMPLETE",zipname,"SHA256",digest,
          "RESULT",test_score["bacc_cell_mean"],flush=True)


if __name__=="__main__":
    main()
