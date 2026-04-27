#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Dict,Tuple
import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import differential_evolution,least_squares
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from config import TARGET_COL
from data_preprocessing import load_raw_data,clean_data,engineer_features
from aci_calculator import compute_aci_predictions

RESULTSS_DIR=ROOT/'resultss'; MODELS_DIR=RESULTSS_DIR/'models'; EQ_DIR=RESULTSS_DIR/'equations'; LOG_DIR=RESULTSS_DIR/'logs'
for d in (MODELS_DIR,EQ_DIR,LOG_DIR): d.mkdir(parents=True,exist_ok=True)

def setup_logger():
    logger.remove(); logger.add(sys.stderr,level='INFO',format='<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}')
    logger.add(str(LOG_DIR/'run_log_closed_form_physics.txt'),level='DEBUG')

def find_col(df:pd.DataFrame,names:list[str])->np.ndarray:
    for n in names:
        if n in df.columns: return df[n].to_numpy(float)
    norm={c.lower().replace(' ','').strip():c for c in df.columns}
    for n in names:
        key=n.lower().replace(' ','').strip()
        if key in norm: return df[norm[key]].to_numpy(float)
    for n in names:
        key=n.lower().replace(' ','').strip()
        hits=[c for c in df.columns if key in c.lower().replace(' ','')]
        if hits:
            logger.warning(f"Column {n!r} not found exactly; using {hits[0]!r}")
            return df[hits[0]].to_numpy(float)
    raise KeyError('Missing columns. Tried: '+str(names)+' Available: '+str(list(df.columns)))

def safe_mape(y,yhat):
    den=np.maximum(np.abs(y),max(float(np.mean(np.abs(y)))*0.05,1.0))
    return float(np.mean(np.clip(np.abs((y-yhat)/den),0,5))*100)

def mets(y,yhat):
    return {'R2':float(r2_score(y,yhat)),'RMSE':float(np.sqrt(mean_squared_error(y,yhat))),'MAE':float(mean_absolute_error(y,yhat)),'MAPE':safe_mape(y,yhat)}

def softplus(z):
    z=np.asarray(z); return np.log1p(np.exp(-np.abs(z)))+np.maximum(z,0)+1e-8

def prepare()->Tuple[np.ndarray,np.ndarray,Dict[str,np.ndarray],Dict[str,float]]:
    logger.info('Loading data and computing ACI baseline')
    df=engineer_features(clean_data(load_raw_data()))
    y=df[TARGET_COL].to_numpy(float)
    maci=np.maximum(compute_aci_predictions(df)['MACI_pred'].to_numpy(float),1e-9)
    eta_pct=find_col(df,['Mass Loss (Tensile bars), ηm (%)','Mass Loss (Tensile bars), eta_m (%)','Mass Loss','eta_m','eta'])
    rho_pct=find_col(df,['Tension Reinforcement Ratio, pten (%)','pten (%)','rho_t','rho'])
    d=find_col(df,['Depth (mm)','d (mm)','depth'])
    b=find_col(df,['Width (mm)','b (mm)','width'])
    fc=find_col(df,["f'c (MPa)",'fc (MPa)','fc'])
    fy=find_col(df,['fy Longitudinal Bars (Tensile), (MPa) ','fy Longitudinal Bars (Tensile), (MPa)','fy (MPa)','fy'])
    db=find_col(df,['Diameter Tensile Bars, db,t (mm)','db,t (mm)','db_t','db'])
    eta=np.clip(eta_pct/100.0,0,0.95); rho=np.clip(rho_pct/100.0,1e-9,None)
    lam=np.clip(d/np.maximum(b,1e-9),1e-9,None); delta=np.clip(db/np.maximum(d,1e-9),1e-9,None)
    phi=np.clip(fy/np.maximum(fc,1e-9),1e-9,None)
    med={'rho_med':float(np.median(rho)),'lam_med':float(np.median(lam)),'delta_med':float(np.median(delta)),'phi_med':float(np.median(phi))}
    X={'eta':eta,'rho_n':rho/max(med['rho_med'],1e-9),'lam_n':lam/max(med['lam_med'],1e-9),'delta_n':delta/max(med['delta_med'],1e-9),'phi_n':phi/max(med['phi_med'],1e-9)}
    logger.info(f'n={len(y)} medians={med}')
    return y,maci,X,med

def subX(X,idx): return {k:v[idx] for k,v in X.items()}

def ratio(th,X):
    a0,a1,a2,a3,a4,b0,b1,b2,b3,b4=th
    rho_lam=X['rho_n']*X['lam_n']; eta=X['eta']
    alpha=softplus(a0+a1*X['phi_n']+a2*rho_lam+a3*X['delta_n']+a4*X['lam_n'])
    beta=softplus(b0+b1*X['phi_n']+b2*rho_lam+b3*X['delta_n']+b4*X['lam_n'])
    R=np.maximum(1-eta,1e-9)**alpha*np.exp(-eta*beta)
    return np.clip(R,0,2.5)

def pred(th,X,maci): return np.maximum(ratio(th,X)*maci,0)

def resid(th,X,y,maci,l2=1e-3):
    yhat=pred(th,X,maci); scale=np.maximum(np.abs(y),max(float(np.mean(np.abs(y)))*0.05,1.0))
    return np.r_[(yhat-y)/scale, np.sqrt(l2)*th]

def fit_one(X,y,maci,seed,fast=False):
    bounds=[(-4,4)]*10
    def obj(th):
        r=resid(th,X,y,maci); return float(np.mean(np.minimum(r*r,4)))
    mi=250 if fast else 800; ps=12 if fast else 25
    logger.info(f'Global optimization DE maxiter={mi} popsize={ps}')
    de=differential_evolution(obj,bounds,seed=seed,maxiter=mi,popsize=ps,tol=1e-7,polish=False,workers=1)
    logger.info('Local robust least_squares')
    lo=np.array([b[0] for b in bounds],float); hi=np.array([b[1] for b in bounds],float)
    ls=least_squares(lambda t:resid(t,X,y,maci),de.x,bounds=(lo,hi),loss='soft_l1',f_scale=0.20,max_nfev=20000)
    return ls.x.astype(float)

def kfold(X,y,maci,k,seed,fast):
    rows=[]; cv=KFold(n_splits=k,shuffle=True,random_state=seed)
    for i,(tr,te) in enumerate(cv.split(y),1):
        th=fit_one(subX(X,tr),y[tr],maci[tr],seed+i,fast)
        mt=mets(y[te],pred(th,subX(X,te),maci[te])); mr=mets(y[tr],pred(th,subX(X,tr),maci[tr]))
        rows.append({'fold':i,'train':mr,'test':mt,'theta':th.tolist()})
        logger.info(f"Fold {i}: test R2={mt['R2']:.4f} RMSE={mt['RMSE']:.4f} MAE={mt['MAE']:.4f} MAPE={mt['MAPE']:.2f}%")
    return rows

def phys(th,X):
    G={k:np.full(200,float(np.median(v))) for k,v in X.items()}; G['eta']=np.linspace(0,0.95,200)
    r=ratio(th,G); return {'R_eta0':float(r[0]),'R_eta095':float(r[-1]),'monotonic_decreasing':bool(np.all(np.diff(r)<=1e-10)),'non_negative':bool(np.all(r>=-1e-12)),'max_ratio_grid':float(np.max(r))}

def save(th,X,y,maci,med,rows):
    full=mets(y,pred(th,X,maci)); chk=phys(th,X)
    keys=['R2','RMSE','MAE','MAPE']; mean={k:float(np.mean([r['test'][k] for r in rows])) for k in keys}; std={k:float(np.std([r['test'][k] for r in rows],ddof=1)) if len(rows)>1 else 0.0 for k in keys}
    out={'approach':'physics_constrained_closed_form_ACI_correction','equation_family':'M_pred=M_ACI*(1-eta)^alpha*exp(-eta*beta)','theta':[float(x) for x in th],'normalization_medians':med,'full_data_metrics':full,'kfold_test_mean':mean,'kfold_test_std':std,'physics_checks':chk,'kfold_details':rows}
    (MODELS_DIR/'closed_form_physics_metrics.json').write_text(json.dumps(out,indent=2))
    a0,a1,a2,a3,a4,b0,b1,b2,b3,b4=th
    txt=f'''# Physics-constrained closed-form corrosion equation

M_pred = M_ACI * R_c
R_c = (1 - eta)^alpha * exp(-eta * beta)

alpha = softplus({a0:.8g} + ({a1:.8g})*phi_n + ({a2:.8g})*(rho_n*lambda_n) + ({a3:.8g})*delta_n + ({a4:.8g})*lambda_n)
beta  = softplus({b0:.8g} + ({b1:.8g})*phi_n + ({b2:.8g})*(rho_n*lambda_n) + ({b3:.8g})*delta_n + ({b4:.8g})*lambda_n)
softplus(z) = ln(1 + exp(z))

Definitions:
eta      = mass_loss_percent / 100
rho_n    = (rho_tension_percent/100) / {med['rho_med']:.10g}
lambda_n = (d / b) / {med['lam_med']:.10g}
delta_n  = (db_t / d) / {med['delta_med']:.10g}
phi_n    = (fy / fc) / {med['phi_med']:.10g}

Full data:
R2={full['R2']:.6f} RMSE={full['RMSE']:.6f} MAE={full['MAE']:.6f} MAPE={full['MAPE']:.3f}%

K-fold test mean:
R2={mean['R2']:.6f} +/- {std['R2']:.6f}
RMSE={mean['RMSE']:.6f} +/- {std['RMSE']:.6f}
MAE={mean['MAE']:.6f} +/- {std['MAE']:.6f}
MAPE={mean['MAPE']:.3f} +/- {std['MAPE']:.3f}%

Physics checks:
R(eta=0)={chk['R_eta0']:.8f}
R(eta=0.95)={chk['R_eta095']:.8f}
monotonic_decreasing={chk['monotonic_decreasing']}
non_negative={chk['non_negative']}
'''
    (EQ_DIR/'closed_form_physics_equation.txt').write_text(txt)
    logger.success(f"Saved equation -> {EQ_DIR/'closed_form_physics_equation.txt'}")
    logger.success(f"Saved metrics  -> {MODELS_DIR/'closed_form_physics_metrics.json'}")

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=42); p.add_argument('--kfold',type=int,default=5); p.add_argument('--fast',action='store_true')
    args=p.parse_args(); setup_logger(); y,maci,X,med=prepare()
    rows=kfold(X,y,maci,args.kfold,args.seed,args.fast)
    logger.info('Fitting final equation on all data')
    th=fit_one(X,y,maci,args.seed+999,args.fast)
    save(th,X,y,maci,med,rows); logger.success('Done')
if __name__=='__main__': main()
