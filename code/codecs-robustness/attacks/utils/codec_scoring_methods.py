import numpy as np
import pandas as pd

# Deltas
# main delta score FR(SRC, RECONSTR. SRC) - FR(ATT, RECONSTR. ATT)  --- CALCULATED W/ DEFENCE
def fr_delta_score(df, fr_method, eps=1e-6, lower_better=False):
    source_vals = np.array(df[f'{fr_method}_clear_defended-rec-clear'])
    dist_vals = np.array(df[f'{fr_method}_attacked_defended-rec-attacked'])
    scores = source_vals - dist_vals
    if lower_better:
        scores *= -1
    scores = scores[~pd.isnull(scores)]
    return np.mean(scores)

# main delta score NR(RECONSTR. SRC) - NR(RECONSTR. ATT)  --- CALCULATED WITH DEFENCE IF PROVIDED
def nr_delta_score(df, nr_method, eps=1e-6, lower_better=False):
    source_vals = np.array(df[f'{nr_method}_defended-rec-clear'])
    dist_vals = np.array(df[f'{nr_method}_defended-rec-attacked'])
    scores = source_vals - dist_vals
    if lower_better:
        scores *= -1
    scores = scores[~pd.isnull(scores)]
    return np.mean(scores)

# additional delta score to evaluate defensive preproc: FR(SRC, RECONSTR. SRC W/O DEFENCE) - FR(SRC, RECONSTR. SRC W/ DEFENCE)
def fr_defence_delta_score(df, fr_method, eps=1e-6, lower_better=False):
    source_vals = np.array(df[f'{fr_method}_clear_rec-clear'])
    dist_vals = np.array(df[f'{fr_method}_clear_defended-rec-clear'])
    scores = source_vals - dist_vals
    if lower_better:
        scores *= -1
    scores = scores[~pd.isnull(scores)]
    return np.mean(scores)

# additional delta score to evaluate defensive preproc: NR(RECONSTR. SRC W/O DEFENCE) - NR(RECONSTR. SRC W/ DEFENCE)
def nr_defence_delta_score(df, nr_method, eps=1e-6, lower_better=False):
    source_vals = np.array(df[f'{nr_method}_rec-clear'])
    dist_vals = np.array(df[f'{nr_method}_defended-rec-clear'])
    scores = source_vals - dist_vals
    if lower_better:
        scores *= -1
    scores = scores[~pd.isnull(scores)]
    return np.mean(scores)

# Mean value of FR metric between clear and attacked: mean(FR(clear, attacked))
def mean_fr_clear_attacked(df, fr_method, eps=1e-6, lower_better=False):
    scores = np.array(df[f'{fr_method}_clear_attacked'])
    scores = scores[~pd.isnull(scores)]
    return np.mean(scores)

# Diffence in NR metric on clear and attacked images wo reconstruction: mean[sgn * (NR(clear) - NR(attacked))]
def delta_nr_clear_attacked(df, nr_method, eps=1e-6, lower_better=False):
    scores_clear = np.array(df[f'{nr_method}_clear'])
    scores_attacked = np.array(df[f'{nr_method}_attacked'])
    scores = scores_clear - scores_attacked
    scores = scores[~pd.isnull(scores)]
    if lower_better:
        scores *= -1
    return np.mean(scores)

fr_2_lower_better = {
    'ssim':False,
    'msssim':False,
    'psnr':False,
    'mse':True,
    'mae':True,
    'l_inf':True,
    'vmaf':False
}
nr_2_lower_better = {
    'niqe' : True
}

def calc_scores_codec(df):
    res = pd.DataFrame(columns=['score', 'value'])
    # FRs, Delta score
    for fr_method in fr_2_lower_better.keys():
        method_val = fr_delta_score(df, fr_method=fr_method, lower_better=fr_2_lower_better[fr_method])
        print(fr_method, ' delta : ', method_val)
        res.loc[len(res)] = {'score':f'delta_{fr_method}', 'value': method_val}
    # NRs, Delta score
    for nr_method in nr_2_lower_better.keys():
        method_val = nr_delta_score(df, nr_method=nr_method, lower_better=nr_2_lower_better[nr_method])
        print(nr_method, ' delta : ', method_val)
        res.loc[len(res)] = {'score':f'delta_{nr_method}', 'value': method_val}
    # FRs, Delta score for defence (useless if no defence is provided)
    for fr_method in fr_2_lower_better.keys():
        method_val = fr_defence_delta_score(df, fr_method=fr_method, lower_better=fr_2_lower_better[fr_method])
        print(fr_method, 'defence delta : ', method_val)
        res.loc[len(res)] = {'score':f'defence_delta_{fr_method}', 'value': method_val}
    # NRs, Delta score for defence (useless if no defence is provided)
    for nr_method in nr_2_lower_better.keys():
        method_val = nr_defence_delta_score(df, nr_method=nr_method, lower_better=nr_2_lower_better[nr_method])
        print(nr_method, 'defence delta : ', method_val)
        res.loc[len(res)] = {'score':f'defence_delta_{nr_method}', 'value': method_val}
    # FRs between clear and attacked
    for fr_method in fr_2_lower_better.keys():
        method_val = mean_fr_clear_attacked(df, fr_method=fr_method, lower_better=fr_2_lower_better[fr_method])
        print('Mean ', fr_method, ' (clear vs attacked) : ', method_val)
        res.loc[len(res)] = {'score':f'mean_{fr_method}', 'value': method_val}
    # Delta between NR on clear and attacked
    for nr_method in nr_2_lower_better.keys():
        method_val = delta_nr_clear_attacked(df, nr_method=nr_method, lower_better=nr_2_lower_better[nr_method])
        print('Mean Delta ', nr_method, ' (clear vs attacked) : ', method_val)
        res.loc[len(res)] = {'score':f'mean_{nr_method}', 'value': method_val}

    for bpp_type in ['defended-clear', 'undefended-clear', 'defended-attacked', 'undefended-attacked']:
        bpp_col = df[f'bpp_{bpp_type}']
        mean_bpp = bpp_col[~np.isnan(bpp_col)].mean()
        print(f'Mean BPP {bpp_type}: ', mean_bpp)
        res.loc[len(res)] = {'score':f'mean_bpp_{bpp_type}', 'value': mean_bpp}
    return res