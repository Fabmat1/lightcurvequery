#!/usr/bin/env python3

import os
import re
import numpy as np
import pandas as pd
from itertools import chain
from astropy.table import Table
from astropy.io import fits
# from fitsio import FITS
try:
    import fitsio
except ImportError as e:
    print(e)
    print("> WARNING: failed to import fitsio; continuing anyway")
try:
    import mesa_reader as mr
except ImportError as e:
    print(e)
    print("> WARNING: failed to import mesa_reader; continuing anyway")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_hezams():
    '''
    He-ZAMS from Paczynski (1971) for Z=0.03 with neutrino emission included
    '''
    d = dict()
    d['mass'] = np.array([0.5, 0.7, 0.85, 1.0, 1.5, 2, 4, 8, 16])
    d['log_mass'] = np.log10(d['mass'])
    d['log_L'] = np.log10([1.53e1, 6.27e1, 1.31e2, 2.36e2, 9.28e2,
                           2.29e3, 1.61e4, 8.00e4, 3.01e5])
    d['log_Teff'] = np.array([4.548, 4.628, 4.671, 4.707, 4.788,
                              4.842, 4.959, 5.059, 5.115])
    d['log_R'] = np.log10([0.105, 0.147, 0.174, 0.198, 0.270,
                           0.331, 0.512, 0.757, 1.080])
    d['R'] = 10**d['log_R']
    d['Teff'] = 10**d['log_Teff']
    d['L'] = 10**d['log_L']
    # M = g * R**2 / G -> g = M*G/R**2; R = sqrt(M*G/g)
    # R = sqrt(mass*1.3271244e+26/pow(10,logg))/6.957e+10
    Const_GMsun_cgs = 1.3271244e+26
    Const_Rsun_cgs = 6.957e+10
    d['log_g'] = np.log10( Const_GMsun_cgs*d['mass'] / \
                          (Const_Rsun_cgs*10**d['log_R'] )**2 )
    d['initial_mass'] = float('nan')
    d['initial_z'] = 0.03
    d['source'] = 'paczynski'
    return AttrDict(d)

def read_yu2021(fpath, plot=False):
    cnames = ["ID", "star_age", "mass", "log_Teff", "log_L", "log_g", "log_R",
              "centerHe", "centerlogT", "Ccoremass"]
    df = pd.read_csv(fpath, sep=",", names=cnames, skiprows=1)
    star_mass = df.loc[len(df)-1,"mass"]
    initial_mass = df.loc[0,"mass"]

    core_he_start = df.loc[0,"centerHe"]
#    df["post_core_he_burning"] = np.where(df["Ccoremass"]>0.01, True, False)
    df["post_core_he_burning"] = np.where(df["centerHe"]<0.0005, True, False)
#    df["post_core_he_burning"] = np.where(df["centerHe"]<0.00001, True, False)
    df["pre_core_he_burning"] = np.where((df["centerlogT"]<8.07) & \
                                         (~df["post_core_he_burning"]),
                                         True, False)
    df["core_he_burning"] = ~(df["pre_core_he_burning"] |\
                              df["post_core_he_burning"])
#    print(df["star_age"].copy()[df["pre_core_he_burning"]] / 1e6)
#    print(df["star_age"].copy()[df["core_he_burning"]] / 1e6)
#    print(df["star_age"].copy()[df["post_core_he_burning"]] / 1e6)
    # remove He flashes except for one model
    rmpre_he = False
#    if not (("030025" in fpath)): rmpre_he = True
    if rmpre_he:
        df = df[~df["pre_core_he_burning"]]
    # remove all steps after Teff = 60000K is reached
    rmhot = False
    if rmhot:
        idx_max = np.argmax(df["log_Teff"]>np.log10(60000))
        df = df.iloc[:idx_max]
    if plot:
        mpre = df["pre_core_he_burning"]
        mpost = df["post_core_he_burning"]
        mhe = df["core_he_burning"]
        ax = plt.gca()
        ax2 = ax.twinx()
        ax.set_xlabel("age / Myr")
        c1 = "log_L"
        c1 = "centerHe"
        c1 = "centerlogT"
#        c1 = "Ccoremass"
        ax.plot(df["star_age"]/1e6, df[c1], color="gray")
        ax2.plot(df["star_age"]/1e6, 10**df["log_Teff"], color="orange")
        ax.plot(df[mhe]["star_age"]/1e6, df[mhe][c1], color="black")
        ax2.plot(df[mpre]["star_age"]/1e6, 10**df[mpre]["log_Teff"], color="green")
        ax2.plot(df[mhe]["star_age"]/1e6, 10**df[mhe]["log_Teff"], color="red")
        ax2.plot(df[mpost]["star_age"]/1e6, 10**df[mpost]["log_Teff"], color="blue")
        ax.set_ylabel(c1)
        ax2.set_ylabel("Teff / K", color="red")
        plt.tight_layout()
        plt.show()

    adict = AttrDict(df)
    if "model" in fpath:
        zstr = fpath.split("/")
        z = float(["0."+s.replace("model", "") for s in zstr if "model" in s][0])
    else:
        z = 0.01
    adict['initial_z'] = z
    adict['source'] = 'yu2021'
    adict['fpath'] = fpath

    fname = os.path.basename(fpath).replace(".csv", "")
#    mass = sum([float(i)/10 for i in mstr])
    adict['star_mass'] = [star_mass]
    adict['initial_mass'] = initial_mass

    return adict

def read_han(fpath, Yc=[0,1], Henv=[0,5e-3], mass=[0,1]):
    '''
    tracks from Han et al. (XX)
    '''
    cnames = ['model_nr', 'mass', 'mass_Henv', 'nr', 'star_age', 'Yc', 'Teff', 'log_g']
    df = pd.read_csv(fpath, sep="\s+", comment='#', names=cnames)

    # mask central helium content
    mask_yc = np.where((df['Yc']>=min(Yc))&(df['Yc']<=max(Yc)), True, False)
    df = df[mask_yc]
    # mask hydrogen envelope mass
    mask_Henv = np.where((df['mass_Henv']>=min(Henv))&(df['mass_Henv']<=max(Henv)), True, False)
    df = df[mask_Henv]
    # mask total mass
    mask_mass = np.where((df['mass']>=min(mass))&(df['mass']<=max(mass)), True, False)
    df = df[mask_mass]

    df['Teff'] = df['Teff']*1.e3
    df['log_Teff'] = np.log10(df['Teff'])
    # M = g * R**2 / G -> R = (M*G / 10**logg)**0.5
    Const_GMsun_cgs = 1.3271244e+26
    Const_Rsun_cgs = 6.957e+10
    df['log_R'] = np.log10(np.sqrt(Const_GMsun_cgs*df['mass']/10**df['log_g'])/Const_Rsun_cgs)
    # L/L_sun = (R/Rsun)^2*(Teff/Const_Teffsun)^4
    Const_Teffsun = 5772.0
    df['log_L'] = 2*df['log_R'] + 4*np.log10(df['Teff']/Const_Teffsun)

    mnr_unique = np.unique(df['model_nr'])
    tracks = []
    for mnr in mnr_unique:
        d = df[df['model_nr']==mnr]
        adict = AttrDict(d)
        adict['initial_mass'] = d['mass'].iloc[0]
        adict['star_mass'] = [d['mass'].iloc[-1]]
        adict['source'] = 'han'
        adict['initial_z'] = 0.014
        adict['mass_Henv'] = d['mass_Henv'].iloc[0]
        Yc_thres = 0.
        adict["post_core_he_burning"] = np.where(d["Yc"]<=Yc_thres, True, False)
        adict["core_he_burning"] = np.where(d["Yc"]>Yc_thres, True, False)
        adict["pre_core_he_burning"] = np.array([False]*len(d["Yc"]))
        adict['fpath'] = fpath
        tracks.append(adict)

    return tracks

def read_michaud2011(fpath):
    pass

def read_zhang_jeffery_2012(fpath, plot=False):
#    print("> reading", fpath)
    cnames = ["star_age", "cocore", "log_L", "log_Teff", "log_R", "log_g"]
    df = pd.read_csv(fpath, sep="\s+", names=cnames, skiprows=1)

    # cut pre-merger
    mask_age = df["star_age"] > 1.6e4 # years
    df = df[mask_age]

    # cut WD stage
    cocore_end = df.iloc[-1]["cocore"]
    mask_co = cocore_end - df["cocore"] > 0.02
    df = df[mask_co]

    if plot:
        mask_teff = 10**df["log_Teff"] < 60000
        ax = plt.gca()
        ax2 = ax.twinx()
        ax.set_xlabel("age / Myr")
        c1 = "log_L"
#        c1 = "cocore"
        ax.plot(df[mask_teff]["star_age"]/1e6, df[mask_teff][c1], color="black")
        ax2.plot(df[mask_teff]["star_age"]/1e6, 10**df[mask_teff]["log_Teff"], color="red")
        ax.set_ylabel(c1)
        ax2.set_ylabel("Teff / K", color="red")
        plt.tight_layout()
        plt.show()

    adict = AttrDict(df)

    adict['initial_z'] = 0.02
    adict['source'] = 'zhang_jeffery_2012'
    adict['fpath'] = fpath

    fname = os.path.basename(fpath).replace(".txt", "")
    mstr = fname.split("p")
    mass = sum([float(i)/10 for i in mstr])
    adict['star_mass'] = [mass]
    adict['initial_mass'] = mass

    return adict


def read_istrate2016(fpath, cut_flashes=False, cut_age=None, cut_Lnuc=False, age_range=False):
    basename = os.path.basename(fpath)
    z = float(basename.split("_")[1])
    cnames = ["model_number", "star_age", "star_mass", "mass_transfer_rate",
              "orbital_period", "neutron_star_mass", "omega", "vrot",
              "ProtPorb1", "log_Teff", "log_L", "log_R", "log_g", "MHec",
              "Menv", "logLnuc", "logLpp", "logLcno", "surfH1", "surfHe4",
              "surfC12", "surfO16", "surfCa40", "MtotH1", "MtotHe4", "logTc",
              "logrhoc"]
    df = pd.read_csv(fpath, sep="\s+", names=cnames)
    mask_transfer = np.where(df["mass_transfer_rate"]<-98, True, False)
    df = df[mask_transfer]

    if cut_flashes:
        Lstart = 10**df["log_L"].iloc[0]
        mask_dL = np.where(abs(dL)/Lstart<=1e-8, True, False)
        df = df[mask_dL]
    if cut_Lnuc:
        mask_dL = np.where(df["logLnuc"]>0, True, False)
        df = df[mask_dL]
#        print(df["star_age"]-df["star_age"][0])
    if cut_age is not None:
        # start from first time Teff = 22000 K is reached
        idx_min = np.argmax(df["log_Teff"]>np.log10(cut_age[0]))
        df = df.iloc[idx_min:]
        mask_age = np.where((df["star_age"]-df.iloc[0]["star_age"])<=cut_age[1], True, False)
        df = df[mask_age]
    if age_range:
        age_start = np.min(df["star_age"])
        ages = df["star_age"] - age_start
        mask_age = np.where((ages > min(age_range)) & \
                            (ages < max(age_range)), True, False)
        df = df[mask_age].copy()

    adict = AttrDict(df)
    adict['initial_mass'] = df['star_mass'].iloc[0]
    adict['initial_z'] = z
    adict['source'] = 'istrate2016'
    adict['fpath'] = fpath
    adict['star_mass'] = [df['star_mass'].iloc[-1]]

    return adict


def read_isis(fpath, istart=99, istop=-1, vequa=0):
    '''
    read track in isis format
    '''
    rename_dict = {"log_teff": "log_Teff",
                   "log_l": "log_L",
                   "logg": "log_g",
                   "radius": "R"}
#    fits = fitsio.FITS(fpath)
    with FITS(fpath) as hdul:
#    with fits.open(fpath) as hdul:
        n_ext = len(hdul)
#        print(n_ext)
        tracks = []
        source = 'isis'
        for i in range(1,n_ext):
            h = hdul[i].read_header()
#            h = hdul[i].header
            extname = h['extname']
            extname = extname.replace("_", "")
            # find keys for dict: -> 1+ connceted letters {1,}
            keys = re.findall('[A-Za-z]{1,}', extname)
            # split on keys
            values = [float(v) for v in re.split('[A-Za-z]{1,}', extname) if v]
            # create dictionary
            pdict = dict(zip(keys, values))
            Z = pdict.get('Z', 0.014)

            d = hdul[i].read()
            # avoid error: Big-endian buffer not supported on little-endian compiler
            d = d.byteswap().newbyteorder()
            df = pd.DataFrame(d)
#            df = Table(hdul[i].data).to_pandas()
            df = df.rename(columns=rename_dict)
            df_cols = df.columns.tolist()
#            print(df)

            if 'mass' in df_cols:
                initial_mass = df['mass'].iloc[0]
                star_mass = df['mass'].iloc[-1]
            else:
                initial_mass = pdict.get('Mi', None)
                star_mass = pdict.get('Mi', None)

            imin = min(df.index)
            imax = max(df.index)
            if (istart<imin) or (istart>imax): istart = imin
            if (istop>imax) or (istop<imin): istop = imax
            if not 'BaSTI_tracks_FEH' in fpath:
                df = df.iloc[istart:istop]
            if 'BaSTI_tracks_MS' in fpath:
                source = 'BaSTI_MS'
            elif 'BaSTI_tracks_FEH' in fpath:
                source = 'BaSTI_HB'
            elif 'MIST' in fpath:
                source = 'MIST'
            if ("vequa" in df_cols) and (vequa is not None):
                cond_vequa = lambda x: x <= vequa
                mask_vequa = cond_vequa(np.array(df["vequa"]))
                if np.sum(mask_vequa) == 0:
                    continue
                else:
                    df = df[mask_vequa].copy()
            df['log_R'] = np.log10(df["R"])

            adict = AttrDict(df)
            adict['source'] = source
            adict['fpath'] = fpath
            adict['initial_z'] = Z
            adict['initial_mass'] = initial_mass
            adict['star_mass'] = [star_mass]
            if "mass" in adict:
                adict["log_mass"] = np.log10(adict["mass"])
            tracks.append(adict)

    tracks = np.array(tracks)
    return tracks


def read_saiojeffery(fpath, Yc=[0,1], plot=False):
    '''
    tracks from Saio & Jeffery (2000)
    https://ui.adsabs.harvard.edu/abs/2000MNRAS.313..671S/abstract
    '''
#    print("> reading", fpath)
    if 'mi03' in fpath:
        columns = ['NSTG', 'star_age', 'log_L', 'log_R', 'log_Teff', 'mass', 'LN', 'log_Tc', 'log_rhoc']
    if 'mi04' in fpath:
        columns = ['NSTG', 'star_age', 'log_L', 'log_R', 'log_Teff', 'mass', 'LN', 'Mshell', 'Yc', 'log_Tc', 'log_rhoc']
    df = pd.read_csv(fpath, sep="\s+", comment="#", names=columns)
    # cut off very cool part
    df = df[2000:]
    # only before core He-buring exhaustion
    cut_postHeMS = False
#    if 'mi04' in fpath: cut_postHeMS = True
    if cut_postHeMS:
        df = df[(df['Yc']>=min(Yc)) & (df['Yc']<=max(Yc))]
    # M = g * R**2 / G -> g = M*G/R**2
    Const_GMsun_cgs = 1.3271244e+26
    Const_Rsun_cgs = 6.957e+10
    df['log_g'] = np.log10( Const_GMsun_cgs*df['mass'] / \
                          (Const_Rsun_cgs*10**df['log_R'] )**2 )

    if plot:
        mask_teff = 10**df["log_Teff"] < 60000
        ax = plt.gca()
        ax2 = ax.twinx()
        ax.set_xlabel("age / Myr")
        c2 = "Yc"
        if c2 in columns:
            ax.plot(df[mask_teff]["star_age"]/1e6, df[mask_teff][c2], color="black")
            ax.set_ylabel(c2)
        ax2.plot(df[mask_teff]["star_age"]/1e6, 10**df[mask_teff]["log_Teff"], color="red")
        ax2.set_ylabel("Teff / K", color="red")
        plt.tight_layout()
        plt.show()

    adict = AttrDict(df)
    adict['source'] = 'saiojeffery'
    adict['initial_z'] = 0.02
    adict['initial_mass'] = df['mass'].iloc[0]
    adict['star_mass'] = [df['mass'].iloc[-1]]
    adict['fpath'] = fpath

    return adict

def read_althaus_2009(fpath):
    def get_lines(fp, line_numbers):
        return tuple(x for i, x in enumerate(fp) if i in line_numbers)
    cnames = ["log_L", "log_Teff", "T_c", "Rho_c", "12Cc","16Oc",
              "star_age", "log_LHe", "log_Lnu", "log_g", "R"]
    df = pd.read_csv(fpath, sep="\s+", comment="#", names=cnames)
    with open(fpath) as fp:
        lines = get_lines(fp, [2])
        lines = [l.strip() for l in lines]
        mass = re.search(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', lines[0])
        mass = float(mass[0])
    adict = AttrDict(df)
    adict['source'] = 'althaus_2009'
    adict['initial_z'] = 0.014
    adict['initial_mass'] = mass
    adict['star_mass'] = mass
    print(adict.log_L)
    return adict

def read_pagb_marcello(fpath):
    cnames = ['N', 'log_L', 'log_Teff', 'log_g',
              'star_age', 'Menv_Mstar', 'mass', 'mdot']
    df = pd.read_csv(fpath, sep="\s+", comment="#", names=cnames)
    # M = g * R**2 / G -> R = (M*G / 10**logg)**0.5
    Const_GMsun_cgs = 1.3271244e+26
    Const_Rsun_cgs = 6.957e+10
    df['log_R'] = np.log10(np.sqrt(Const_GMsun_cgs*df['mass']/10**df['log_g'])/Const_Rsun_cgs)
    adict = AttrDict(df)
    adict['source'] = 'pagb'
    adict['initial_z'] = 0.014
    adict['initial_mass'] = df['mass'].iloc[0]
    adict['star_mass'] = [df['mass'].iloc[-1]]
    adict['fpath'] = fpath

    return adict

def read_dorman(fpath, Yc=[0,1], M_ini=False, M_env=False):
    with open(fpath, 'r') as f:
        lines = f.read().splitlines()
    nlines = len(lines)
    ifirst = []
    for i in range(nlines):
        if 'Track:' in lines[i]:
            ifirst.append(i)
    ntab = len(ifirst)
    cnames = ['star_age', 'Yc', 'log_Teff', 'log_L', 'log_g', 'log_R', 'Msh', 'log_Tc', 'log_rc']
    tracks = []
    for i in range(ntab):
        tab_header = lines[ifirst[i]].split()
        mass = float(tab_header[5])
        Fe_H = float(tab_header[8])
        O_Fe = float(tab_header[11])
        Y = float(tab_header[14])
        istart = ifirst[i]+2
        if i < ntab-1:
            istop = ifirst[i+1]
        else:
            istop = nlines
        lines_tab = lines[istart:istop]
        data = np.genfromtxt(lines_tab)
        dataT = data
        data = np.transpose(data)
        mask_yc = np.where((data[1]>=min(Yc))&(data[1]<=max(Yc)), True, False)
        data = np.transpose(dataT[mask_yc])
        d = dict(zip(cnames, data))
        d['initial_mass'] = mass
        d['star_mass'] = [mass]
        d['log_star_mass'] = [mass]
        d['source'] = 'dorman'
        d['Menv'] = mass - d['Msh']
        d['Menv_max'] = np.max(d['Menv'])
        d['initial_z'] = 10**Fe_H * 0.014
        Const_Rsun_cgs = 6.957e+10
        d['log_R'] = np.log10(10**np.array(d['log_R'])/Const_Rsun_cgs)
        d['R'] = 10**d['log_R']
        d['Teff'] = 10**d['log_Teff']
        d['L'] = 10**d['log_L']
        d['fpath'] = fpath
        d["post_core_he_burning"] = np.where(d["Yc"]<0.0005, True, False)
        d["pre_core_he_burning"] = np.where((d["log_Tc"]<8.07) & \
                                            (~d["post_core_he_burning"]),
                                            True, False)
        d["core_he_burning"] = ~(d["pre_core_he_burning"] |\
                                 d["post_core_he_burning"])
        tracks.append(AttrDict(d))

    if M_ini:
        mask_tracks = [(t.initial_mass >= min(M_ini)) and (t.initial_mass <= max(M_ini)) for t in tracks]
        tracks = np.array(tracks)[mask_tracks]
    if M_env:
        menv_max = np.array([t.Menv_max for t in tracks])
        mask_tracks = np.where((menv_max > min(M_env)) & (menv_max < max(M_env)), True, False)
        tracks = np.array(tracks)[mask_tracks]

    return tracks

def fill_tracks(fp_track1, fp_track2,
                xc="log_Teff", yc="log_L",
                plot=False):
    if "pagb" in fp_track1:
        t1 = read_pagb_marcello(fp_track1)
    elif "althaus_2009" in fp_track1:
        t1 = read_althaus_2009(fp_track1)
    else:
        t1 = fp_track1
    if "pagb" in fp_track2:
        t2 = read_pagb_marcello(fp_track2)
    elif "althaus_2009" in fp_track2:
        t2 = read_althaus_2009(fp_track2)
    else:
        t2 = fp_track2
    x1 = t1[xc]
    x2 = t2[xc]
    y1 = t1[yc]
    y2 = t2[yc]
    if plot:
        temp_fig, temp_ax = plt.subplots()
        temp_ax.plot(x1, y1, color="black")
        temp_ax.plot(x2, y2, color="red")
        plt.show()
#    yt = np.array([y1,y2[::-1]]).flatten()
#    xt = np.array([x1,x2[::-1]]).flatten()
    yt = np.concatenate([y1,y2[::-1]])
    xt = np.concatenate([x1,x2[::-1]])
#    plt.fill(xt, yt)
    return xt, yt

def create_isochrone_from_tracks(hs, age_isochrone=0.01, filter_source=False):
    if filter_source:
        iMS = [hs[i]['source'] == filter_source for i in range(len(hs))]
        hs_isochrone = hs[iMS]
    else:
        hs_isochrone = hs
#    filter_vequa = False
#    if filter_vequa:
#        vequa_thres = 0.1
#        iMS = [np.where(hs[i]['vequa'] <= vequa_thres, True, False) for i in range(len(hs_isochrone))]
#        hs_isochrone = [hs_isochrone[i][iMS[i]] for i in range(len(hs_isochrone))]
    # sort by mass
    if "initial_mass" in hs_isochrone[0].keys():
        iMS = np.argsort([hs_isochrone[i]['initial_mass'] for i in range(len(hs_isochrone))])
        hs_isochrone = hs_isochrone[iMS]
    isochrone_dict = {"log_Teff":[], "log_L":[], "log_g":[], "R":[], "mass":[]}
    for i in range(len(hs_isochrone)):
        age_isochrone_itp = age_isochrone
        if age_isochrone_itp > max(hs_isochrone[i]['age']):
            age_isochrone_itp = max(hs_isochrone[i]['age'])
        if age_isochrone_itp < min(hs_isochrone[i]['age']):
            age_isochrone_itp = min(hs_isochrone[i]['age'])
        for key in isochrone_dict.keys():
            xy_intp = np.interp(age_isochrone_itp, hs_isochrone[i]['age'], hs_isochrone[i][key])
            isochrone_dict[key].append(xy_intp)
    isochrone_dict.update({"log_R": np.log10(isochrone_dict["R"])})
    isochrone_dict = AttrDict(isochrone_dict)
    isochrone_dict['source'] = hs[0]['source']
    isochrone_dict['fpath'] = "None"
    isochrone_dict['initial_z'] = 0
    isochrone_dict['initial_mass'] = None
    isochrone_dict['star_mass'] = [0]

    return isochrone_dict


def create_sequence_from_point(hs, point, filter_source=False):
    if type(hs) == str:
        if "dorman" in hs:
            hs = read_dorman(hs, Yc=[0.001,1.00])
        else:
            hs = read_isis(hs)
    hs = np.array(hs)
    # filter_source="BaSTI_MS" BaSTI_HB
    if filter_source:
        sources = [hs[i]['source'] for i in range(len(hs))]
        if not filter_source in sources:
            print("> %s not found" % filter_source)
        iMS = [hs[i]['source'] == filter_source for i in range(len(hs))]
        hs_isochrone = hs[iMS]
    else:
        hs_isochrone = hs
    # sort by mass
    iMS = np.argsort([hs_isochrone[i]['star_mass'][0] for i in range(len(hs_isochrone))])
    hs_isochrone = hs_isochrone[iMS]

    isochrone_dict = {"log_Teff":[], "log_L":[], "log_g":[], "R":[], "mass":[], "age":[]}
    npoint = len(hs_isochrone[0]["log_Teff"])
    for i in range(len(hs_isochrone)):
        for key in isochrone_dict.keys():
            if type(point)==int:
                xy_intp = hs_isochrone[i][key][point]
            else:
                xy_intp = np.interp(point, np.arange(npoint), hs_isochrone[i][key])
            isochrone_dict[key].append(xy_intp)
    isochrone_dict.update({"imass":[]})
    for i in range(len(hs_isochrone)):
        isochrone_dict["imass"].append(hs_isochrone[i]["initial_mass"])
#        isochrone_dict["mass"].append(hs_isochrone[i]["mass"])
    for key in isochrone_dict:
        isochrone_dict[key] = np.array(isochrone_dict[key])
    isochrone_dict.update({"log_R": np.log10(isochrone_dict["R"])})
    isochrone_dict = AttrDict(isochrone_dict)
    isochrone_dict['source'] = hs[0]['source']
    isochrone_dict['fpath'] = "None"
    isochrone_dict['initial_z'] = 0
    isochrone_dict['initial_mass'] = None
    isochrone_dict['star_mass'] = [0]
    return isochrone_dict

def fill_isis_isochrone(fp_track="/userdata/data/dorsch/isis/stellar_isisscripts/refdata/BaSTI_tracks_MSlow.fits.gz",
                        xc="log_Teff", yc="log_L",
                        age1=0., age2=200000, # Myr
                        z1=2.000e-05, z2=0.01721,
                        M_min=0.03, M_max=15.,
                        points=False, plot=False):
    # =2.000e-05 -> -3.20; 1.258e-02 -> -0.1; 0.01721 -> +0.06
    zthres = 1e-7

    istart = 99
    istop = -1
    if (type(points) == str) or (type(points) == list):
        if type(points) == str:
            if points == "MS":
                points = [100, 360]
            elif points == "RGB":
                points = [490, 1290]
                points = [360, 1290]
            elif points == "MS_RGB":
                points = [100, 1290]
        points = np.array(points)
        istart = min(points[np.where(points>0,True,False)])
        if min(points) < 0:
            istop = min(points)
        else:
            istop = max(points)
    hd = read_isis(fp_track, istart=istart, istop=istop)
    h1 = [h for h in hd if (h['initial_mass']>=M_min and h['initial_mass']<=M_max and \
                            h['initial_z'] > z1-zthres and h['initial_z'] < z1+zthres)]
    isort = np.argsort([h["initial_mass"] for h in h1])
    h1 = np.array(h1)[isort]
    t1 = create_isochrone_from_tracks(h1, age_isochrone=age1)
    x1 = t1[xc]
    y1 = t1[yc]
    if (z1 != z2) or (age2 != age1):
        h2 = [h for h in hd if (h['initial_mass']>=M_min and h['initial_mass']<=M_max and \
                                h['initial_z'] > z2-zthres and h['initial_z'] < z2+zthres)]
        isort = np.argsort([h["initial_mass"] for h in h2])
        h2 = np.array(h2)[isort]
        t2 = create_isochrone_from_tracks(h2, age_isochrone=age2)

        print_seq = False
        if print_seq:
            print_squence([t1, t2], ["log_Teff", "log_L", "log_g", "log_R", "mass"])

        y2 = t2[yc]
        x2 = t2[xc]
        yt = np.array([y1, y2[::-1]]).flatten()
        xt = np.array([x1, x2[::-1]]).flatten()
    else:
        yt = y1
        xt = x1
    if plot:
        temp_fig, temp_ax = plt.subplots()
        temp_ax.plot(x1, y1, color="black")
        if (z1 != z2) or (age2 != age1):
            temp_ax.plot(x2, y2, color="red")
        plt.show()

    return xt, yt

def fill_basti(fpath="/userdata/data/dorsch/isis/stellar_isisscripts/refdata/BaSTI_tracks_MShigh.fits.gz",
               xc="log_Teff", yc="log_L", Z=1.258e-02):
    # low Z: Z=0.00002 (-> -2.8)
    # solar Z: Z=1.258e-02 (-> Z=-0.05)
    if type(fpath) == str:
        hd = read_isis(fpath)
    else:
        hd = fpath
    zthres = 1e-7
    hd = [h for h in hd if (h['initial_mass']>0.03 and h['initial_mass']<15.0 and \
                            h['initial_z'] > Z-zthres and h['initial_z'] < Z+zthres)]
    isort = np.argsort([h["initial_mass"] for h in hd])
    hd = np.array(hd)[isort]
    x_left = [list(h[xc])[0] for h in hd]
    y_left = [list(h[yc])[0] for h in hd]
    x_right = [list(h[xc])[-1] for h in hd]
    y_right = [list(h[yc])[-1] for h in hd]
    x_top = list(hd[-1][xc])
    y_top = list(hd[-1][yc])
    x_bottom = list(hd[0][xc])
    y_bottom = list(hd[0][yc])
    if yc=="log_L" or yc=="log_R":
        x = list(chain.from_iterable([x_left, x_top, x_right[::-1], x_bottom[::-1]]))
        y = list(chain.from_iterable([y_left, y_top, y_right[::-1], y_bottom[::-1]]))
    elif yc=="log_g":
        x = list(chain.from_iterable([x_left, x_top, x_bottom[::-1]]))
        y = list(chain.from_iterable([y_left, y_top, y_bottom[::-1]]))

    return x, y

def fill_dorman(fpath="e64.dat",
                xc="log_Teff", yc="log_g"):
    """
    z22.dat # [Fe/H] =  0
    j63.dat # [Fe/H] = -0.47
    e64.dat # [Fe/H] = -1.48
    """
    hd = read_dorman(fpath, Yc=[0.001,1.00])
    isort = np.argsort([h["initial_mass"] for h in hd])
    hd = np.array(hd)[isort]
    x_bottom = [list(h[xc])[0] for h in hd]
    x_top = [list(h[xc])[-1] for h in hd]
    apply_ylog = False
    if "mass" in yc:
        if "log" in yc:
            apply_ylog = True
        yc = yc.replace("mass", "star_mass")
#        y_bottom = [h["initial_mass"] for h in hd]
#        y_top = np.array(y_bottom) # + 0.01
    y_bottom = [list(h[yc])[0] for h in hd]
    y_top = [list(h[yc])[-1] for h in hd]
    # dorman is too short (cool)
    if not "mass" in yc:
        x_bottom_add = [4.53]
        x_top_add = [4.59]
        if yc=="log_L":
            y_bottom_add = [1.07]
            y_top_add = [1.492]
        elif yc=="log_R":
            y_bottom_add = [-1.006]
            y_top_add = [-0.903]
        elif yc=="log_g":
            y_bottom_add = [6.125]
            y_top_add = [6.02]

        x_bottom = x_bottom_add + x_bottom
        y_bottom = y_bottom_add + y_bottom
        x_top = x_top_add + x_top
        y_top = y_top_add + y_top

    x = list(chain.from_iterable([x_bottom, x_top[::-1]]))
    y = list(chain.from_iterable([y_bottom, y_top[::-1]]))
    if apply_ylog: y = np.log10(y)

    return x, y

def read_hs(paths):
    hs = []
    for path in paths:
        if 'history.data' in path:
            try:
                h = mr.MesaData(path)
                hs.append(h)
                print("> read %s" % path)
            except:
                print("> could not read %s" % path)
        else:
            if 'dorman' in path:
#                hd = read_dorman(path,
#                                 Yc=[0,1],
#                                 M_ini=[0.2, 0.8],
#                                 M_env=[0, 0.01])
                hd = read_dorman(path,
                                 Yc=[0,1],
                                 M_ini=[0.2, 0.8],
                                 M_env=[0, 0.003]) # [0.001,1]
            elif 'saiojeffery' in path:
                hd = read_saiojeffery(path)
                hd = [hd]
            elif 'zhang_jeffery' in path:
                hd = [read_zhang_jeffery_2012(path)]
            elif 'pagb' in path:
                hd = read_pagb_marcello(path)
                hd = [hd]
            elif 'alina' in path:
                cut_age = [6400, 1e9] # Teff_0, age after Teff_0 is first reached
                cut_age = [8400, 1e9]
                cut_age = None
                cut_flashes = False
                age_range = [0, 799.5e6]
                hd = read_istrate2016(path, cut_age=cut_age,
                                      cut_flashes=cut_flashes, age_range=age_range)
                hd = [hd]
            elif 'sdbevol' in path:
#                hd = read_han(path,
#                              Yc=[1e-10,1], # [1e-10,1]
#                              Henv=[0,5e-3], # [0,5e-3], [2e-4,4e-3]
#                              mass=[0,0.51]) # [0,0.51]
                hd = read_han(path,
                              Yc=[0,1], # [1e-10,1]
                              Henv=[1e-3,4e-3], # [0,5e-3], [2e-4,4e-3]
                              mass=[0.45,0.49]) # [0,0.51]
            elif 'yu2021' in path:
                hd = read_yu2021(path)
                hd = [hd]
            elif 'refdata' in path:
                istop = -1
                istart = 100
                if 'MShigh' in path:
                    istart = 100 # 1290
                    istop = 1950
                hd = read_isis(path, istart=istart, istop=istop, vequa=0)
                if np.ndim(hd) == 0:
                    hd = [hd]
            hs.extend(hd)
            print("> read %s" % path)
    return hs
