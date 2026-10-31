"""Independent science-and-contract audit of exact BCI Track 02 ZIP.

All tests use a fixed seed and only public test artifacts. No access to
private/hidden challenge answers. No changes to the trained model.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

ROOT=Path(__file__).resolve().parent
KIT=Path(__import__("os").environ.get("BCI_KIT_DIR",str(ROOT/"2026-competition")))/"tracks"/"bci_decoding"
ZIP=ROOT/"Agnuxo_SpectralBCI_Track02.zip"
OUT=ROOT/"audit_replay_latest.json"
SEED=20261031
sys.path.insert(0,str(KIT))
ZIP=ROOT/"Agnuxo_SpectralBCI_Track02.zip"
OUT=ROOT/"audit_replay_latest.json"
SEED=20261031
rng=np.random.default_rng(SEED)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


with tempfile.TemporaryDirectory(prefix="track02_audit_") as d:
    with zipfile.ZipFile(ZIP) as a:
        members=sorted(a.namelist())
        assert members==["submission.py","weights.npz"],members
        assert a.testzip() is None
        a.extractall(d)
    src=Path(d)
    spec=importlib.util.spec_from_file_location("packed_neural_bci",src/"submission.py")
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    with np.load(src/"weights.npz",allow_pickle=False) as weights:
        w={k:weights[k].copy() for k in weights.files}
    t1={
      "name":"T1_exact_zip_contract_and_weight_safety",
      "sha256":digest(ZIP),
      "members":members,
      "classes":w["classes"].astype(int).tolist(),
      "feature_dimension":int(w["center"].shape[0]),
      "finite":bool(all(np.isfinite(v).all() for v in w.values())),
      "coefficient_shape":list(w["coef"].shape),
    }
    t1["pass"]=bool(
      t1["finite"] and t1["classes"]==[0,1,2] and
      w["coef"].shape==(3,492) and w["scale"].shape==(492,)
    )
    m=module.SpectralBCIModel(w,41,480,120)
    X=rng.normal(0,35,size=(30,41,480)).astype(np.float32)
    p_all=m.predict(X)
    p_ind=np.asarray([int(m.predict(x[None])[0]) for x in X])
    t2={"name":"T2_batch_vs_single_stream_prediction_equivalence",
        "windows":len(X),"disagreements":int((p_all!=p_ind).sum()),
        "pass":bool(np.array_equal(p_all,p_ind))}
    X2=X.copy()
    X2[11:]+=rng.normal(0,90,size=X2[11:].shape)
    p2=m.predict(X2)
    t3={"name":"T3_no_future_window_contamination",
        "prefix_windows":11,
        "disagreements":int((p_all[:11]!=p2[:11]).sum()),
        "pass":bool(np.array_equal(p_all[:11],p2[:11]))}
    t4={"name":"T4_finite_class_prediction_on_malformed_signals",
        "all_finite_and_in_class":False,"exceptions":[]}
    cases=[np.zeros((1,41,480),np.float32),
           np.full((1,41,480),np.nan,np.float32),
           np.full((1,41,480),np.inf,np.float32),
           np.full((1,41,480),1e10,np.float32)]
    for i,c in enumerate(cases):
        try:
            y=m.predict(c)
            if y.shape!=(1,) or int(y[0]) not in (0,1,2):
                t4["exceptions"].append(f"Case {i}: unexpected output {y}")
        except Exception as exc:
            t4["exceptions"].append(f"Case {i}: {exc!r}")
    t4["all_finite_and_in_class"]=not bool(t4["exceptions"])
    t4["pass"]=t4["all_finite_and_in_class"]

    tests=[t1,t2,t3,t4]
    if (ROOT/"test_features.npz").exists():
        meta=json.loads((ROOT/"training_results.json").read_text(encoding="utf-8"))
        with np.load(ROOT/"test_features.npz",allow_pickle=False) as z:
            test={k:z[k].copy() for k in z.files}
        z=(test["X"]-w["center"])/np.maximum(w["scale"],1e-5)
        predicted=w["classes"][np.argmax(z@w["coef"].T+w["intercept"],axis=1)]
        y=test["y"]
        def cellscores(ytrue,ypred):
            g=np.stack([test["subject"],test["stream"],test["context"]],axis=1)
            return np.array([balanced_accuracy_score(ytrue[np.all(g==c,axis=1)],
                                                      ypred[np.all(g==c,axis=1)])
                             for c in np.unique(g,axis=0)])
        scores=cellscores(y,predicted)
        ci=np.percentile(rng.choice(scores,size=(2000,len(scores)),replace=True).mean(axis=1),[2.5,97.5])
        t5={"name":"T5_independent_cell_level_recalculation_with_cluster_bootstrap",
            "n_cells":len(scores),
            "cell_scores":scores.round(6).tolist(),
            "mean_score":float(scores.mean()),
            "training_log_score":float(meta["heldout_test"]["bacc_cell_mean"]),
            "bootstrap_mean_95_ci":[float(ci[0]),float(ci[1])],
            "random_class_floor":1/3,
            "pass":bool(abs(scores.mean()-meta["heldout_test"]["bacc_cell_mean"])<1e-10)}
        tests.append(t5)
    else:
        print("T5 not repeated: public held-out feature cache is absent. Original five-test report preserved in independent_20261031.json.",flush=True)
    full={"seed":SEED,"zip_sha256":digest(ZIP),
          "tests":tests,
          "pass":all(t["pass"] for t in tests),
          "note":"No private test data used. Bootstrap is over eight public subject×session×context cells; CI does not establish official leaderboard rank."}
    OUT.write_text(json.dumps(full,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(full,ensure_ascii=False,indent=2),flush=True)
    if not full["pass"]:
        raise SystemExit(1)
