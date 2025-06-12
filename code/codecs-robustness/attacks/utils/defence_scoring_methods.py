import numpy as np
import pandas as pd
from scipy import stats
# util funcs for gains
def rel_gain(src_vals, atk_vals, eps=1e-6):
    return np.mean(np.abs(src_vals - atk_vals) / (src_vals + eps))
def abs_gain(src_vals, atk_vals, metric_range=1):
    return np.mean(np.abs(src_vals - atk_vals) / metric_range)
def srocc(src_vals, other_vals):
    try:
        return stats.spearmanr(src_vals, other_vals).statistic
    except:
        rho, pval = stats.spearmanr(src_vals, other_vals)
        return rho


# Rel/Abs gain after defence is applied
def robust_rel_gain(df, metric_range=1, eps=1e-6):
    ref_vals = np.array(df['clear'])
    other_vals = np.array(df['defended-attacked'])
    return rel_gain(ref_vals, other_vals, eps=eps)

def robust_abs_gain(df, metric_range=1, eps=1e-6):
    ref_vals = np.array(df['clear'])
    other_vals = np.array(df['defended-attacked'])
    return abs_gain(ref_vals, other_vals, metric_range=metric_range)

# Rel/Abs gain after defence is applied to BOTH clear and attacked
def both_defended_rel_gain(df, metric_range=1, eps=1e-6):
    ref_vals = np.array(df['defended-clear'])
    other_vals = np.array(df['defended-attacked'])
    return rel_gain(ref_vals, other_vals, eps=eps)

def both_defended_abs_gain(df, metric_range=1, eps=1e-6):
    ref_vals = np.array(df['defended-clear'])
    other_vals = np.array(df['defended-attacked'])
    return abs_gain(ref_vals, other_vals, metric_range=metric_range)

# Rel/Abs gain on unpurified (before defence) attacked images
def nonpurified_rel_gain(df, metric_range=1, eps=1e-6):
    ref_vals = np.array(df['clear'])
    other_vals = np.array(df['attacked'])
    return rel_gain(ref_vals, other_vals, eps=eps)

def nonpurified_abs_gain(df, metric_range=1, eps=1e-6):
    ref_vals = np.array(df['clear'])
    other_vals = np.array(df['attacked'])
    return abs_gain(ref_vals, other_vals, metric_range=metric_range)

# SSIM/PSNR score of purified attacked images
def defence_similarity_score(df, metric_range=1, eps=1e-6):
    ssim_vals = np.array(df['ssim_clear_defended-attacked'])
    psnr_vals = np.array(df['psnr_clear_defended-attacked'])
    scores = ssim_vals  + psnr_vals / 40
    scores = scores[~np.isnan(scores)]
    return np.mean(scores)

# SSIM/PSNR score of purified clear images
def defence_clear_similarity_score(df, metric_range=1, eps=1e-6):
    ssim_vals = np.array(df['ssim_clear_defended-clear'])
    psnr_vals = np.array(df['psnr_clear_defended-clear'])
    scores = ssim_vals + psnr_vals / 40
    scores = scores[~np.isnan(scores)]
    return np.mean(scores)

# SROCC between MOS(on clear images) and metric values on PURIFIED attacked images
def robust_attacked_srocc_mos(df, metric_range=1, eps=1e-6):
    if 'mos' not in df.columns:
        return np.nan
    ref_vals = np.array(df['mos'])
    other_vals = np.array(df['defended-attacked'])
    return srocc(ref_vals, other_vals)

# SROCC between MOS(on clear images) and metric values on PURIFIED clear images
def robust_clear_srocc_mos(df, metric_range=1, eps=1e-6):
    if 'mos' not in df.columns:
        return np.nan
    ref_vals = np.array(df['mos'])
    other_vals = np.array(df['defended-clear'])
    return srocc(ref_vals, other_vals)

# SROCC between MOS(on clear images) and metric values on NON-PURIFIED clear images
def clear_srocc_mos(df, metric_range=1, eps=1e-6):
    if 'mos' not in df.columns:
        return np.nan
    ref_vals = np.array(df['mos'])
    other_vals = np.array(df['clear'])
    return srocc(ref_vals, other_vals)

# SROCC between MOS(on clear images) and metric values on NON-PURIFIED attacked images
def attacked_srocc_mos(df, metric_range=1, eps=1e-6):
    if 'mos' not in df.columns:
        return np.nan
    ref_vals = np.array(df['mos'])
    other_vals = np.array(df['attacked'])
    return srocc(ref_vals, other_vals)

# SROCC between metric values before and after purification on clear, non-attacked images
def robust_clear_srocc_clear(df, metric_range=1, eps=1e-6):
    ref_vals = np.array(df['clear'])
    other_vals = np.array(df['defended-clear'])
    return srocc(ref_vals, other_vals)


defence_method_name_to_func = {
            #gains
            'robust_rel_gain': robust_rel_gain,
            'robust_abs_gain':robust_abs_gain,
            'nonpurified_rel_gain':nonpurified_rel_gain,
            'nonpurified_abs_gain':nonpurified_abs_gain,
            'both_defended_rel_gain':both_defended_rel_gain,
            'both_defended_abs_gain':both_defended_abs_gain,
            #SSIM/PSNR score
            'similarity_score':defence_similarity_score,
            'clear_similarity_score':defence_clear_similarity_score,
            #DIfferent SROCCs
            'robust_attacked_srocc_mos':robust_attacked_srocc_mos,
            'robust_clear_srocc_mos':robust_clear_srocc_mos,
            'clear_srocc_mos':clear_srocc_mos,
            'attacked_srocc_mos':attacked_srocc_mos,
            'robust_clear_srocc_clear':robust_clear_srocc_clear,
}


def calc_scores_defence(df, metric_range=1):
    res = pd.DataFrame(columns=['score', 'value'])
    for method_name in defence_method_name_to_func.keys():
        method_val = defence_method_name_to_func[method_name](df, metric_range=metric_range)
        print(method_name, ' : ', method_val)
        res.loc[len(res)] = {'score':method_name, 'value': method_val}
    return res
