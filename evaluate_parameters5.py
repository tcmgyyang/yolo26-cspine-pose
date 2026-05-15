"""
3.5 Accuracy Evaluation of Parameter Calculation — FINAL
=========================================================
- 7 static + 4 dynamic cervical spine radiographic parameters
- Cross-folder (test+val) scanning for maximum sample size
- ICC(2,1), Pearson, Bland-Altman, measurement time comparison
- Publication-quality figures grouped by clinical category:
    Fig1: Sagittal Alignment (Cobb, Lordosis Index, cSVA)
    Fig2: Vertebral Morphometry (DHR, Slip, DHI, Canal)
    Fig3: Dynamic Function (Seg ROM, Global ROM, Seg Trans, ISD Change)
    Fig4: ICC Forest Plot (all 11 parameters overview)

Usage: python evaluate_parameters.py
"""

import warnings
warnings.filterwarnings('ignore')
import os, sys, time
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches

# ============================================================
# Configuration
# ============================================================
WEIGHTS_PATH = 'runs/pose/spine_extreme/DPCA/DPCA_AC_sigma/weights/best.pt'
DATA_ROOT    = 'dataset'
OUTPUT_DIR   = 'runs/pose/spine_extreme/parameter_evaluation'
IMGSZ        = 1280
CONF_THRESH  = 0.5
MANUAL_TIME_PER_IMAGE = 120.0  # seconds; replace with real timing data if available

SCAN_DIRS = [
    os.path.join(DATA_ROOT, 'images', 'test'),
    os.path.join(DATA_ROOT, 'images', 'val'),
]

# ============================================================
# Plot Style
# ============================================================
CLR = {
    'dot_face':'#4A90D9','dot_edge':'#1A5276',
    'bias':'#C0392B','bias_band':'#F5B7B1',
    'loa_line':'#27AE60','loa_fill':'#D5F5E3',
    'identity':'#7F8C8D','zero_ref':'#BDC3C7',
    'pred_fill':'#D6EAF8','pred_edge':'#85C1E9',
    'title':'#1B2631','label':'#2C3E50','annot':'#566573',
    'bg':'#FDFEFE','grid':'#EAECEE',
    'cat1':'#2E86C1','cat2':'#8E44AD','cat3':'#D35400',
}

plt.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':9,'axes.labelsize':10,'axes.titlesize':11,
    'axes.titleweight':'bold','axes.linewidth':0.9,
    'xtick.labelsize':8,'ytick.labelsize':8,
    'xtick.direction':'in','ytick.direction':'in',
    'xtick.major.size':4,'ytick.major.size':4,
    'xtick.minor.visible':True,'ytick.minor.visible':True,
    'xtick.minor.size':2,'ytick.minor.size':2,
    'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
})

# ============================================================
# Keypoint Mapping (35 keypoints)
# ============================================================
VERTEBRAE = {
    'C2':{'AS':0,'AI':1,'PS':2,'PI':3,'SP':4},
    'C3':{'AS':5,'AI':6,'PS':7,'PI':8,'SP':9,'LP':10},
    'C4':{'AS':11,'AI':12,'PS':13,'PI':14,'SP':15,'LP':16},
    'C5':{'AS':17,'AI':18,'PS':19,'PI':20,'SP':21,'LP':22},
    'C6':{'AS':23,'AI':24,'PS':25,'PI':26,'SP':27,'LP':28},
    'C7':{'AS':29,'AI':30,'PS':31,'PI':32,'SP':33,'LP':34},
}

# ============================================================
# Geometry Utilities
# ============================================================
def pt(kps,i): return np.array([kps[i][0],kps[i][1]])
def mid(a,b): return (a+b)/2.0
def edist(a,b): return np.linalg.norm(a-b)
def angle2v(v1,v2):
    c=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-12)
    return np.degrees(np.arccos(np.clip(c,-1,1)))
def perp_dist(point,lp1,lp2):
    lv=lp2-lp1; ll=np.linalg.norm(lv)
    if ll<1e-12: return np.linalg.norm(point-lp1)
    lu=lv/ll; pp=lp1+np.dot(point-lp1,lu)*lu
    return np.linalg.norm(point-pp)

# ============================================================
# 7 Static Parameters
# ============================================================
def calc_cobb_angle(k):
    return angle2v(pt(k,3)-pt(k,1), pt(k,32)-pt(k,30))

def calc_lordosis_index(k):
    p2,p7=pt(k,3),pt(k,32); D=edist(p2,p7)
    if D<1e-6: return 0.0
    return max(perp_dist(pt(k,i),p2,p7) for i in [8,14,20,26])/D*100

def calc_csva(k):
    c2c=(pt(k,0)+pt(k,1)+pt(k,2)+pt(k,3))/4.0
    return c2c[0]-pt(k,31)[0]

def calc_dhr(k):
    rs=[]
    for v in ['C3','C4','C5','C6','C7']:
        d=VERTEBRAE[v]
        ah=edist(pt(k,d['AS']),pt(k,d['AI'])); ph=edist(pt(k,d['PS']),pt(k,d['PI']))
        h=(ah+ph)/2; dm=edist(mid(pt(k,d['AS']),pt(k,d['AI'])),mid(pt(k,d['PS']),pt(k,d['PI'])))
        if h>1e-6: rs.append(dm/h)
    return np.mean(rs) if rs else 0.0

def calc_slip(k):
    segs=[(3,7),(8,13),(14,19),(20,25),(26,31)]
    return max(abs(pt(k,a)[0]-pt(k,b)[0]) for a,b in segs)

def calc_dhi(k):
    segs=[('C2','C3'),('C3','C4'),('C4','C5'),('C5','C6'),('C6','C7')]
    ds=[]
    for u,l in segs:
        U,L=VERTEBRAE[u],VERTEBRAE[l]
        dh=(edist(pt(k,U['AI']),pt(k,L['AS']))+edist(pt(k,U['PI']),pt(k,L['PS'])))/2
        bh=(edist(pt(k,L['AS']),pt(k,L['AI']))+edist(pt(k,L['PS']),pt(k,L['PI'])))/2
        if bh>1e-6: ds.append(dh/bh)
    return np.mean(ds) if ds else 0.0

def calc_canal(k):
    ds=[]
    for v in ['C3','C4','C5','C6','C7']:
        d=VERTEBRAE[v]
        if 'LP' not in d: continue
        ds.append(edist(mid(pt(k,d['PS']),pt(k,d['PI'])),pt(k,d['LP'])))
    return np.mean(ds) if ds else 0.0

STATIC_PARAMS = [
    ('Cobb_Angle',calc_cobb_angle),
    ('Lordosis_Index',calc_lordosis_index),
    ('cSVA',calc_csva),
    ('Diameter_Height_Ratio',calc_dhr),
    ('Vertebral_Slip',calc_slip),
    ('Disc_Height_Index',calc_dhi),
    ('Spinal_Canal_Diameter',calc_canal),
]

# ============================================================
# 4 Dynamic Parameters
# ============================================================
def calc_seg_rom(kf,ke):
    segs=[(1,3,6,8),(6,8,12,14),(12,14,18,20),(18,20,24,26),(24,26,30,32)]
    return np.mean([abs(angle2v(pt(kf,b)-pt(kf,a),pt(kf,d)-pt(kf,c))-
                        angle2v(pt(ke,b)-pt(ke,a),pt(ke,d)-pt(ke,c))) for a,b,c,d in segs])

def calc_global_rom(kf,ke):
    return abs(calc_cobb_angle(kf)-calc_cobb_angle(ke))

def calc_seg_trans(kf,ke):
    return max(abs(pt(kf,i)[0]-pt(ke,i)[0]) for i in [3,8,14,20,26,32])

def calc_isd_change(kf,ke):
    sp=[4,9,15,21,27,33]
    return np.mean([edist(pt(kf,sp[i]),pt(kf,sp[i+1]))-edist(pt(ke,sp[i]),pt(ke,sp[i+1]))
                     for i in range(len(sp)-1)])

DYNAMIC_PARAMS = [
    ('Segmental_ROM',calc_seg_rom),
    ('Global_ROM',calc_global_rom),
    ('Segmental_Translation',calc_seg_trans),
    ('Interspinous_Distance_Change',calc_isd_change),
]

# Category grouping for figures
PARAM_CATEGORIES = {
    'Sagittal Alignment Parameters': {
        'color': CLR['cat1'],
        'keys': ['Cobb_Angle','Lordosis_Index','cSVA'],
        'display': {
            'Cobb_Angle': 'Cobb Angle (°)',
            'Lordosis_Index': 'Lordosis Index (%)',
            'cSVA': 'cSVA (pixels)',
        },
    },
    'Vertebral Morphometry Parameters': {
        'color': CLR['cat2'],
        'keys': ['Diameter_Height_Ratio','Vertebral_Slip','Disc_Height_Index','Spinal_Canal_Diameter'],
        'display': {
            'Diameter_Height_Ratio': 'Diameter–Height Ratio',
            'Vertebral_Slip': 'Vertebral Slip (pixels)',
            'Disc_Height_Index': 'Disc Height Index',
            'Spinal_Canal_Diameter': 'Spinal Canal Diameter (pixels)',
        },
    },
    'Dynamic Function Parameters': {
        'color': CLR['cat3'],
        'keys': ['Segmental_ROM','Global_ROM','Segmental_Translation','Interspinous_Distance_Change'],
        'display': {
            'Segmental_ROM': 'Segmental ROM (°)',
            'Global_ROM': 'Global ROM (°)',
            'Segmental_Translation': 'Segmental Translation (pixels)',
            'Interspinous_Distance_Change': 'Interspinous Distance Change (pixels)',
        },
    },
}

# ============================================================
# YOLO I/O
# ============================================================
def load_yolo_kps(path,w,h):
    with open(path) as f: lines=f.readlines()
    if not lines: return None
    vals=[float(x) for x in lines[0].strip().split()[5:]]
    if len(vals)<105: return None
    kps=np.zeros((35,3))
    for i in range(35): kps[i]=[vals[i*3]*w, vals[i*3+1]*h, vals[i*3+2]]
    return kps

def get_label_path(img_path):
    """Convert image path to label path.
    Handles both normal (xxx.jpg -> xxx.txt) and double-ext (xxx.jpg.jpg -> xxx.jpg.txt).
    Only strips the LAST image extension, matching YOLO's label naming convention.
    """
    lp = img_path.replace('/images/', '/labels/')
    # Strip only the last extension
    base, ext = os.path.splitext(lp)
    return base + '.txt'

# ============================================================
# Cross-Folder Scanner
# ============================================================
def scan_all(dirs):
    """Scan dirs for:
    - lateral images (with _lateral suffix) + all non-suffixed images (public dataset) -> static params
    - flexion-extension pairs (cross-folder) -> dynamic params
    """
    patients=defaultdict(dict); laterals=[]; public_laterals=[]
    exts={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
    for d in dirs:
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            if not any(f.lower().endswith(e) for e in exts): continue
            full=os.path.join(d,f)
            # Strip only the last extension for name parsing
            name, _ = os.path.splitext(f)
            # If still ends with image ext (double-ext case), strip again
            _, ext2 = os.path.splitext(name)
            if ext2.lower() in exts:
                name, _ = os.path.splitext(name)
            if '_extension' in name:
                patients[name.rsplit('_extension',1)[0]]['extension']=full
            elif '_flexion' in name:
                patients[name.rsplit('_flexion',1)[0]]['flexion']=full
            elif '_lateral' in name:
                patients[name.rsplit('_lateral',1)[0]]['lateral']=full
                laterals.append(full)
            else:
                # Public dataset: no view suffix -> treat as lateral for static params
                public_laterals.append(full)
    pairs={p:v for p,v in patients.items() if 'flexion' in v and 'extension' in v}
    all_laterals = laterals + public_laterals
    return laterals, public_laterals, all_laterals, pairs

# ============================================================
# Statistics
# ============================================================
def compute_icc(ya,ym):
    n=len(ya)
    if n<3: return np.nan,np.nan,np.nan
    data=np.column_stack([ym,ya]); k=2
    gm=np.mean(data); rm=np.mean(data,1); cm=np.mean(data,0)
    sst=np.sum((data-gm)**2)
    ssr=k*np.sum((rm-gm)**2); ssc=n*np.sum((cm-gm)**2)
    sse=sst-ssr-ssc
    msr=ssr/(n-1); msc=ssc/(k-1); mse=sse/((n-1)*(k-1)) if (n-1)*(k-1)>0 else 1e-12
    icc=(msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)
    F=msr/mse if mse>1e-12 else 1e6
    df1,df2=n-1,(n-1)*(k-1)
    FL=F/stats.f.ppf(0.975,df1,df2); FU=F*stats.f.ppf(0.975,df2,df1)
    lo=(FL-1)/(FL+k-1); hi=(FU-1)/(FU+k-1)
    return float(np.clip(icc,-1,1)),float(np.clip(lo,-1,1)),float(np.clip(hi,-1,1))

def icc_interp(v):
    if np.isnan(v): return 'N/A'
    if v<0.50: return 'Poor'
    elif v<0.75: return 'Moderate'
    elif v<0.90: return 'Good'
    else: return 'Excellent'

def ba_stats(ya,ym):
    d=ya-ym; b=np.mean(d); s=np.std(d,ddof=1)
    return b,s,b-1.96*s,b+1.96*s

# ============================================================
# Publication Plot Subfunctions
# ============================================================
def draw_ba(ax, ya, ym, name):
    means=(ya+ym)/2; diffs=ya-ym; n=len(diffs)
    bias=np.mean(diffs); sd=np.std(diffs,ddof=1)
    se=sd/np.sqrt(n); ul=bias+1.96*sd; ll=bias-1.96*sd

    ax.set_facecolor(CLR['bg'])
    ax.grid(True,ls='-',alpha=0.35,color=CLR['grid'],zorder=0)
    xr=[means.min()-0.1*np.ptp(means), means.max()+0.18*np.ptp(means)]
    ax.fill_between(xr,ll,ul,color=CLR['loa_fill'],alpha=0.55,zorder=1)
    ax.fill_between(xr,bias-1.96*se,bias+1.96*se,color=CLR['bias_band'],alpha=0.5,zorder=2)
    for v in [ul,ll]:
        ax.axhline(v,color=CLR['loa_line'],ls='--',lw=1.3,zorder=3,
                   path_effects=[pe.withStroke(linewidth=2.8,foreground='white',alpha=0.3)])
    ax.axhline(bias,color=CLR['bias'],ls='-',lw=1.8,zorder=4,
               path_effects=[pe.withStroke(linewidth=3.5,foreground='white',alpha=0.3)])
    if abs(bias)>0.05*np.ptp(diffs):
        ax.axhline(0,color=CLR['zero_ref'],ls=':',lw=0.8,zorder=2)
    ax.scatter(means,diffs,s=30,alpha=0.72,fc=CLR['dot_face'],ec=CLR['dot_edge'],lw=0.7,zorder=5)

    bb=dict(boxstyle='round,pad=0.3',fc='white',ec='#D5D8DC',alpha=0.93,lw=0.6)
    ax.annotate(f'Mean bias = {bias:.2f}',xy=(xr[1]*0.97,bias),fontsize=8,
                fontweight='bold',color=CLR['bias'],ha='right',va='bottom',bbox=bb)
    ax.annotate(f'+1.96 SD = {ul:.2f}',xy=(xr[1]*0.97,ul),fontsize=7.5,
                color=CLR['loa_line'],ha='right',va='bottom',bbox=bb)
    ax.annotate(f'−1.96 SD = {ll:.2f}',xy=(xr[1]*0.97,ll),fontsize=7.5,
                color=CLR['loa_line'],ha='right',va='top',bbox=bb)
    ax.text(0.03,0.95,f'n = {n}',transform=ax.transAxes,fontsize=8.5,fontweight='bold',
            color=CLR['annot'],va='top',bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='#D5D8DC',lw=0.6))
    ax.set_xlabel('Mean of Automated and Manual',fontsize=9,color=CLR['label'],labelpad=5)
    ax.set_ylabel('Difference (Auto − Manual)',fontsize=9,color=CLR['label'],labelpad=5)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color('#ABB2B9'); ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors='#566573',width=0.6)


def draw_corr(ax, ya, ym, name):
    n=len(ya); r,p=stats.pearsonr(ya,ym); r2=r**2
    sl,ic,_,_,_=stats.linregress(ym,ya)
    xf=np.linspace(min(ym)*0.9,max(ym)*1.1,200); yf=sl*xf+ic
    xm_=np.mean(ym); ssx_=np.sum((ym-xm_)**2)
    res_=ya-(sl*ym+ic); mse_=np.sum(res_**2)/(n-2)
    sep_=np.sqrt(mse_*(1+1/n+(xf-xm_)**2/ssx_))
    sec_=np.sqrt(mse_*(1/n+(xf-xm_)**2/ssx_))
    tc_=stats.t.ppf(0.975,n-2)

    ax.set_facecolor(CLR['bg'])
    ax.grid(True,ls='-',alpha=0.35,color=CLR['grid'],zorder=0)
    ax.fill_between(xf,yf-tc_*sep_,yf+tc_*sep_,color=CLR['pred_fill'],alpha=0.35,zorder=1)
    ax.fill_between(xf,yf-tc_*sec_,yf+tc_*sec_,color=CLR['pred_edge'],alpha=0.3,zorder=2)
    av=np.concatenate([ym,ya])
    lims=[np.min(av)-0.08*np.ptp(av),np.max(av)+0.08*np.ptp(av)]
    ax.plot(lims,lims,color=CLR['identity'],ls='--',lw=1.1,alpha=0.6,zorder=3)
    ax.plot(xf,yf,color=CLR['bias'],lw=2.2,zorder=4,
            path_effects=[pe.withStroke(linewidth=4,foreground='white',alpha=0.4)])
    ax.scatter(ym,ya,s=30,alpha=0.72,fc=CLR['dot_face'],ec=CLR['dot_edge'],lw=0.7,zorder=5)

    ps='p < 0.001' if p<0.001 else f'p = {p:.3f}'
    ax.text(0.04,0.96,f'n = {n}\nr = {r:.4f}\nR² = {r2:.4f}\n{ps}',
            transform=ax.transAxes,fontsize=8.5,va='top',color=CLR['title'],linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.45',fc='white',ec=CLR['dot_edge'],lw=1.0,alpha=0.95))
    sgn='+' if ic>=0 else '−'
    ax.text(0.96,0.06,f'y = {sl:.3f}x {sgn} {abs(ic):.2f}',transform=ax.transAxes,fontsize=8,
            ha='right',color=CLR['annot'],style='italic',
            bbox=dict(boxstyle='round,pad=0.25',fc='white',ec='#D5D8DC',lw=0.5,alpha=0.9))
    ax.set_xlabel('Manual Measurement',fontsize=9,color=CLR['label'],labelpad=5)
    ax.set_ylabel('Automated Measurement',fontsize=9,color=CLR['label'],labelpad=5)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color('#ABB2B9'); ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors='#566573',width=0.6)


def generate_category_figures(all_results, plots_dir):
    """Generate 3 category-based figures (BA left + Corr right)."""
    fig_idx = 1
    for cat_name, cat_cfg in PARAM_CATEGORIES.items():
        keys = cat_cfg['keys']
        display = cat_cfg['display']
        cat_color = cat_cfg['color']

        # Collect available data
        available = []
        for key in keys:
            if key in all_results:
                ya, ym = all_results[key]
                if len(ya) >= 3:
                    available.append((key, display.get(key, key), ya, ym))

        if not available:
            print(f"  Skip {cat_name}: no data")
            continue

        nrows = len(available)
        fig_h = 4.8 * nrows + 1.2
        fig, axes = plt.subplots(nrows, 2, figsize=(16, fig_h))
        fig.patch.set_facecolor('white')
        if nrows == 1:
            axes = axes.reshape(1, -1)

        for i, (key, dname, ya, ym) in enumerate(available):
            letter_ba = chr(97 + i * 2)
            letter_corr = chr(97 + i * 2 + 1)

            draw_ba(axes[i][0], ya, ym, dname)
            axes[i][0].set_title(f'({letter_ba}) {dname}', fontsize=10.5, fontweight='bold',
                                 color=CLR['title'], pad=10, loc='left')

            draw_corr(axes[i][1], ya, ym, dname)
            axes[i][1].set_title(f'({letter_corr}) {dname}', fontsize=10.5, fontweight='bold',
                                 color=CLR['title'], pad=10, loc='left')

        fig.text(0.27, 0.995, 'Bland–Altman Analysis', fontsize=13, fontweight='bold',
                 color='#566573', ha='center', va='top', style='italic')
        fig.text(0.76, 0.995, 'Correlation Analysis', fontsize=13, fontweight='bold',
                 color='#566573', ha='center', va='top', style='italic')

        fig.suptitle(f'Figure {fig_idx}.  {cat_name}',
                     fontsize=15, fontweight='bold', color=cat_color, y=1.02, x=0.5, ha='center')
        line = plt.Line2D([0.1, 0.9], [1.005, 1.005], transform=fig.transFigure,
                           color=cat_color, linewidth=2.5, alpha=0.7)
        fig.add_artist(line)

        plt.tight_layout(rect=[0, 0, 1, 0.99], h_pad=4.0, w_pad=3.5)
        fname = f'Fig{fig_idx}_{cat_name.replace(" ", "_")}'
        fig.savefig(os.path.join(plots_dir, f'{fname}.png'), dpi=300, facecolor='white',
                    bbox_inches='tight', pad_inches=0.3)
        fig.savefig(os.path.join(plots_dir, f'{fname}.pdf'), facecolor='white',
                    bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)
        print(f"  {fname} saved ({nrows} params)")
        fig_idx += 1


def generate_forest_plot(summary, plots_dir):
    """Generate ICC Forest Plot for all 11 parameters."""

    # Build data from summary list
    DISPLAY = {
        'Cobb_Angle': 'Cobb Angle',
        'Lordosis_Index': 'Lordosis Index',
        'cSVA': 'cSVA',
        'Diameter_Height_Ratio': 'Diameter–Height Ratio',
        'Vertebral_Slip': 'Vertebral Slip',
        'Disc_Height_Index': 'Disc Height Index',
        'Spinal_Canal_Diameter': 'Spinal Canal Diameter',
        'Segmental_ROM': 'Segmental ROM',
        'Global_ROM': 'Global ROM',
        'Segmental_Translation': 'Segmental Translation',
        'Interspinous_Distance_Change': 'ISD Change',
    }
    CAT_MAP = {
        'Cobb_Angle': 'Sagittal\nAlignment',
        'Lordosis_Index': 'Sagittal\nAlignment',
        'cSVA': 'Sagittal\nAlignment',
        'Diameter_Height_Ratio': 'Vertebral\nMorphometry',
        'Vertebral_Slip': 'Vertebral\nMorphometry',
        'Disc_Height_Index': 'Vertebral\nMorphometry',
        'Spinal_Canal_Diameter': 'Vertebral\nMorphometry',
        'Segmental_ROM': 'Dynamic\nFunction',
        'Global_ROM': 'Dynamic\nFunction',
        'Segmental_Translation': 'Dynamic\nFunction',
        'Interspinous_Distance_Change': 'Dynamic\nFunction',
    }
    CAT_COLORS = {
        'Sagittal\nAlignment': CLR['cat1'],
        'Vertebral\nMorphometry': CLR['cat2'],
        'Dynamic\nFunction': CLR['cat3'],
    }

    # Ordered param list
    ORDER = ['Cobb_Angle','Lordosis_Index','cSVA',
             'Diameter_Height_Ratio','Vertebral_Slip','Disc_Height_Index','Spinal_Canal_Diameter',
             'Segmental_ROM','Global_ROM','Segmental_Translation','Interspinous_Distance_Change']

    rows = []
    summary_dict = {r['Parameter']: r for r in summary}
    for key in ORDER:
        if key not in summary_dict:
            continue
        r = summary_dict[key]
        rows.append((DISPLAY.get(key, key), CAT_MAP[key],
                      float(r['ICC']), float(r['ICC_Lower']), float(r['ICC_Upper']), int(r['N'])))

    if not rows:
        print("  Forest plot: no data, skipped")
        return

    ZONES = [
        (0.0, 0.50, '#FADBD8', 'Poor'),
        (0.50, 0.75, '#FEF9E7', 'Moderate'),
        (0.75, 0.90, '#E8F8F5', 'Good'),
        (0.90, 1.00, '#D5F5E3', 'Excellent'),
    ]

    n_params = len(rows)
    y_positions = np.arange(n_params, 0, -1)

    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Background zones
    for x_lo, x_hi, color, label in ZONES:
        ax.axvspan(x_lo, x_hi, color=color, alpha=0.5, zorder=0)
    for x_lo, x_hi, color, label in ZONES:
        ax.text((x_lo + x_hi) / 2, n_params + 0.75, label,
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='#566573', style='italic')
    for thresh in [0.50, 0.75, 0.90]:
        ax.axvline(thresh, color='#ABB2B9', ls=':', lw=0.8, zorder=1)

    # Category separators
    cat_boundaries = []
    prev_cat = None
    for i, (_, cat, *_) in enumerate(rows):
        if cat != prev_cat and prev_cat is not None:
            cat_boundaries.append(i)
        prev_cat = cat
    for b in cat_boundaries:
        ax.axhline(y_positions[b] + 0.5, color='#D5D8DC', ls='-', lw=0.8, zorder=1)

    # Plot each parameter
    cat_y_ranges = {}
    for i, (name, cat, icc, ci_lo, ci_hi, n) in enumerate(rows):
        y = y_positions[i]
        color = CAT_COLORS[cat]

        ax.plot([ci_lo, ci_hi], [y, y], color=color, lw=2.2, solid_capstyle='round', zorder=3)
        cap_h = 0.18
        ax.plot([ci_lo, ci_lo], [y - cap_h, y + cap_h], color=color, lw=1.5, zorder=3)
        ax.plot([ci_hi, ci_hi], [y - cap_h, y + cap_h], color=color, lw=1.5, zorder=3)
        ax.scatter(icc, y, s=90, color=color, edgecolors='white', linewidths=1.2,
                   zorder=4, marker='D')

        ax.text(1.02, y, f'{icc:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]',
                va='center', ha='left', fontsize=9, color='#2C3E50',
                transform=ax.get_yaxis_transform())
        ax.text(1.28, y, f'(n={n})',
                va='center', ha='left', fontsize=8, color='#7F8C8D',
                transform=ax.get_yaxis_transform())

        if cat not in cat_y_ranges:
            cat_y_ranges[cat] = [y, y]
        else:
            cat_y_ranges[cat][0] = min(cat_y_ranges[cat][0], y)
            cat_y_ranges[cat][1] = max(cat_y_ranges[cat][1], y)

    # Category brackets
    for cat, (y_min, y_max) in cat_y_ranges.items():
        y_mid = (y_min + y_max) / 2
        color = CAT_COLORS[cat]
        x_bracket = -0.03
        ax.annotate('', xy=(x_bracket, y_min - 0.3), xytext=(x_bracket, y_max + 0.3),
                    xycoords=('axes fraction', 'data'), textcoords=('axes fraction', 'data'),
                    arrowprops=dict(arrowstyle='-', color=color, lw=2.0))
        ax.plot([x_bracket - 0.008, x_bracket], [y_max + 0.3, y_max + 0.3],
                color=color, lw=2.0, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot([x_bracket - 0.008, x_bracket], [y_min - 0.3, y_min - 0.3],
                color=color, lw=2.0, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.text(x_bracket - 0.015, y_mid, cat,
                va='center', ha='right', fontsize=9, fontweight='bold', color=color,
                transform=ax.get_yaxis_transform(), linespacing=1.2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10, color='#2C3E50')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.2, n_params + 1.2)
    ax.set_xlabel('Intraclass Correlation Coefficient  ICC(2,1)', fontsize=12,
                  color='#2C3E50', labelpad=10)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 0.90, 1.0])
    ax.set_xticklabels(['0', '0.25', '0.50', '0.75', '0.90', '1.0'], fontsize=10)

    ax.set_title('Agreement Between Automated and Manual Measurements:\n'
                 'ICC(2,1) with 95% Confidence Intervals',
                 fontsize=14, fontweight='bold', color='#1B2631', pad=20)

    ax.text(1.02, n_params + 0.75, 'ICC  [95% CI]',
            va='center', ha='left', fontsize=9, fontweight='bold', color='#566573',
            transform=ax.get_yaxis_transform())
    ax.text(1.28, n_params + 0.75, 'N',
            va='center', ha='left', fontsize=9, fontweight='bold', color='#566573',
            transform=ax.get_yaxis_transform())

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ABB2B9')
    ax.spines['bottom'].set_color('#ABB2B9')
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', colors='#566573')

    legend_elements = [mpatches.Patch(facecolor=CAT_COLORS[c], label=c.replace('\n', ' '))
                       for c in CAT_COLORS]
    leg = ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
                    framealpha=0.95, edgecolor='#D5D8DC', fancybox=True,
                    title='Parameter Category', title_fontsize=9)
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'Fig4_ICC_Forest_Plot.png'), dpi=300,
                facecolor='white', bbox_inches='tight', pad_inches=0.4)
    fig.savefig(os.path.join(plots_dir, 'Fig4_ICC_Forest_Plot.pdf'),
                facecolor='white', bbox_inches='tight', pad_inches=0.4)
    plt.close(fig)
    print("  Fig4_ICC_Forest_Plot saved")


# ============================================================
# Main Pipeline
# ============================================================
def main():
    from ultralytics import YOLO
    from PIL import Image

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # ---- Step 1: Scan ----
    print("=" * 70)
    print("Step 1: Scanning test + val (cross-folder matching)")
    print("=" * 70)
    clinical_laterals, public_laterals, all_laterals, pairs = scan_all(SCAN_DIRS)
    print(f"  Clinical lateral images (_lateral suffix): {len(clinical_laterals)}")
    print(f"  Public dataset images (no suffix):         {len(public_laterals)}")
    print(f"  Total for static params:                   {len(all_laterals)}")
    print(f"  Flex-ext pairs for dynamic params:         {len(pairs)}")

    # ---- Step 2: Load model ----
    print(f"\nLoading: {WEIGHTS_PATH}")
    model = YOLO(WEIGHTS_PATH)

    # ---- Step 3: Static parameters ----
    print("\n" + "=" * 70)
    print(f"Step 2: Static parameters (n={len(all_laterals)})")
    print("=" * 70)
    sres = []; stimes = []
    for ip in sorted(all_laterals):
        fn = os.path.basename(ip); lp = get_label_path(ip)
        if not os.path.exists(lp): print(f"  Skip {fn}: no label"); continue
        img = Image.open(ip); w, h = img.size
        gt = load_yolo_kps(lp, w, h)
        if gt is None: print(f"  Skip {fn}: GT fail"); continue
        t0 = time.time()
        res = model.predict(ip, imgsz=IMGSZ, conf=CONF_THRESH, verbose=False)
        dt = time.time() - t0; stimes.append(dt)
        if not res or res[0].keypoints is None or len(res[0].keypoints.data) == 0:
            print(f"  Skip {fn}: no det"); continue
        pk = res[0].keypoints.data[0].cpu().numpy()
        row = {'image': fn, 'time': dt}
        for nm, fc in STATIC_PARAMS:
            try: row[f'{nm}_auto'] = fc(pk); row[f'{nm}_manual'] = fc(gt)
            except: row[f'{nm}_auto'] = np.nan; row[f'{nm}_manual'] = np.nan
        sres.append(row); print(f"  OK: {fn} ({dt:.3f}s)")

    dfs = pd.DataFrame(sres)
    dfs.to_csv(os.path.join(OUTPUT_DIR, 'static_params_raw.csv'), index=False)
    print(f"\nStatic done: {len(dfs)} images")

    # ---- Step 4: Dynamic parameters ----
    print("\n" + "=" * 70)
    print(f"Step 3: Dynamic parameters (n={len(pairs)})")
    print("=" * 70)
    dres = []; dtimes = []
    for pid, views in sorted(pairs.items()):
        fp, ep = views['flexion'], views['extension']
        fl, el = get_label_path(fp), get_label_path(ep)
        if not os.path.exists(fl) or not os.path.exists(el):
            print(f"  Skip {pid}: label missing"); continue
        fi, ei = Image.open(fp), Image.open(ep)
        fw, fh = fi.size; ew, eh = ei.size
        gf = load_yolo_kps(fl, fw, fh); ge = load_yolo_kps(el, ew, eh)
        if gf is None or ge is None: print(f"  Skip {pid}: GT fail"); continue
        t0 = time.time()
        rf = model.predict(fp, imgsz=IMGSZ, conf=CONF_THRESH, verbose=False)
        re = model.predict(ep, imgsz=IMGSZ, conf=CONF_THRESH, verbose=False)
        dt = time.time() - t0; dtimes.append(dt)
        if (not rf or rf[0].keypoints is None or len(rf[0].keypoints.data) == 0 or
                not re or re[0].keypoints is None or len(re[0].keypoints.data) == 0):
            print(f"  Skip {pid}: no det"); continue
        pf = rf[0].keypoints.data[0].cpu().numpy()
        pek = re[0].keypoints.data[0].cpu().numpy()
        row = {'patient': pid, 'time': dt}
        for nm, fc in DYNAMIC_PARAMS:
            try: row[f'{nm}_auto'] = fc(pf, pek); row[f'{nm}_manual'] = fc(gf, ge)
            except: row[f'{nm}_auto'] = np.nan; row[f'{nm}_manual'] = np.nan
        dres.append(row); print(f"  OK: {pid} ({dt:.3f}s)")

    dfd = pd.DataFrame(dres)
    dfd.to_csv(os.path.join(OUTPUT_DIR, 'dynamic_params_raw.csv'), index=False)
    print(f"\nDynamic done: {len(dfd)} pairs")

    # ============================================================
    # Step 5: Statistical Analysis
    # ============================================================
    print("\n" + "=" * 130)
    print("Step 4: Statistical Analysis")
    print("=" * 130)

    summary = []
    all_results = {}  # key -> (ya, ym) for plotting

    def analyze(df, plist, ptype):
        print(f"\n--- {ptype} Parameters ---")
        hdr = f"  {'Parameter':<35} {'N':>4} {'ICC':>8} {'95% CI':<22} {'Interp':<12} {'r':>8} {'Bias':>10} {'LoA':<24}"
        print(hdr); print("  " + "-" * 125)
        for nm, _ in plist:
            ca, cm = f'{nm}_auto', f'{nm}_manual'
            if ca not in df.columns: continue
            v = df[[ca, cm]].dropna()
            ya, ym = v[ca].values.astype(float), v[cm].values.astype(float)
            if len(ya) < 3: print(f"  {nm}: n={len(ya)} skip"); continue

            all_results[nm] = (ya, ym)

            icc, lo, hi = compute_icc(ya, ym); interp = icc_interp(icc)
            r, p = stats.pearsonr(ya, ym)
            bias, sd, llo, lhi = ba_stats(ya, ym)

            print(f"  {nm:<35} {len(ya):>4} {icc:>8.4f} [{lo:.3f}, {hi:.3f}]{'':<6} "
                  f"{interp:<12} {r:>8.4f} {bias:>10.4f} [{llo:.3f}, {lhi:.3f}]")

            summary.append({
                'Parameter': nm, 'Type': ptype, 'N': len(ya),
                'ICC': f'{icc:.4f}', 'ICC_Lower': f'{lo:.4f}', 'ICC_Upper': f'{hi:.4f}',
                'Interpretation': interp,
                'Pearson_r': f'{r:.4f}', 'Pearson_p': f'{p:.2e}',
                'Bias': f'{bias:.4f}', 'SD_Diff': f'{sd:.4f}',
                'LoA_Lower': f'{llo:.4f}', 'LoA_Upper': f'{lhi:.4f}',
            })

    if len(dfs) >= 3: analyze(dfs, STATIC_PARAMS, 'Static')
    if len(dfd) >= 3: analyze(dfd, DYNAMIC_PARAMS, 'Dynamic')

    # ---- Time comparison ----
    print(f"\n--- Measurement Time ---")
    at = np.array(stimes + dtimes)
    if len(at) >= 8:
        mt = np.full_like(at, MANUAL_TIME_PER_IMAGE)
        print(f"  Auto:   {np.mean(at):.3f} ± {np.std(at):.3f} sec")
        print(f"  Manual: {MANUAL_TIME_PER_IMAGE:.1f} sec")
        _, swp = stats.shapiro(at)
        if swp > 0.05:
            ts, tp = stats.ttest_rel(at, mt); print(f"  Paired t: t={ts:.3f}, p={tp:.2e}")
        else:
            ws, wp = stats.wilcoxon(at - mt); print(f"  Wilcoxon: W={ws:.3f}, p={wp:.2e}")

    # ============================================================
    # Step 6: Generate Figures
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 5: Generating publication figures")
    print("=" * 70)
    generate_category_figures(all_results, plots_dir)
    generate_forest_plot(summary, plots_dir)

    # ============================================================
    # Step 7: Save Tables
    # ============================================================
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(OUTPUT_DIR, 'agreement_summary.csv'), index=False)

    with open(os.path.join(OUTPUT_DIR, 'agreement_table_latex.txt'), 'w') as f:
        f.write("Parameter & Type & N & ICC (95\\% CI) & Interp. & Pearson $r$ & Bias & 95\\% LoA \\\\\n")
        f.write("\\hline\n")
        for r in summary:
            ci = f"({r['ICC_Lower']}, {r['ICC_Upper']})"
            loa = f"({r['LoA_Lower']}, {r['LoA_Upper']})"
            f.write(f"{r['Parameter']} & {r['Type']} & {r['N']} & {r['ICC']} {ci} & "
                    f"{r['Interpretation']} & {r['Pearson_r']} & {r['Bias']} & {loa} \\\\\n")

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print(f"  static_params_raw.csv")
    print(f"  dynamic_params_raw.csv")
    print(f"  agreement_summary.csv")
    print(f"  agreement_table_latex.txt")
    print(f"  plots/Fig1_Sagittal_Alignment_Parameters.png/.pdf")
    print(f"  plots/Fig2_Vertebral_Morphometry_Parameters.png/.pdf")
    print(f"  plots/Fig3_Dynamic_Function_Parameters.png/.pdf")
    print(f"  plots/Fig4_ICC_Forest_Plot.png/.pdf")
    print(f"{'=' * 70}")
    print("Done!")


if __name__ == '__main__':
    main()
