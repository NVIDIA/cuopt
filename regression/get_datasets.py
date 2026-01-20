# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import urllib.request
import urllib.parse
import ssl
import subprocess


# From: https://plato.asu.edu/bench.html
# Folder containg instances:
# - https://miplib2010.zib.de/miplib2010.php
# - https://www.netlib.org/lp/data/
# - https://old.sztaki.hu/~meszaros/public_ftp/lptestset/ (and it's subfolders)
# - https://plato.asu.edu/ftp/lptestset/ (and it's subfolders)
# - https://miplib.zib.de/tag_benchmark.html
# - https://miplib.zib.de/tag_collection.html

LPFeasibleMittelmannSet = [
    "L1_sixm250obs",
    "Linf_520c",
    "a2864",
    "bdry2",
    "cont1",
    "cont11",
    "datt256_lp",
    "dlr1",
    "ex10",
    "fhnw-binschedule1",
    "fome13",
    "graph40-40",
    "irish-electricity",
    "neos",
    "neos3",
    "neos-3025225",
    "neos-5052403-cygnet",
    "neos-5251015",
    "ns1687037",
    "ns1688926",
    "nug08-3rd",
    "pds-100",
    "physiciansched3-3",
    "qap15",
    "rail02",
    "rail4284",
    "rmine15",
    "s82",
    "s100",
    "s250r10",
    "savsched1",
    "scpm1",
    "shs1023",
    "square41",
    "stat96v2",
    "stormG2_1000",
    "stp3d",
    "supportcase10",
    "tpl-tub-ws1617",
    "woodlands09",
    "Dual2_5000",
    "Primal2_1000",
    "thk_48",
    "thk_63",
    "L1_sixm1000obs",
    "L2CTA3D",
    "degme",
    "dlr2",
    "set-cover-model",
]

MiplibInstances = [
    "30n20b8.mps",
    "cryptanalysiskb128n5obj14.mps",
    "graph20-20-1rand.mps",
    "n2seq36q.mps",
    "neos-4338804-snowy.mps",
    "neos-957323.mps",
    "rail01.mps",
    "splice1k1.mps",
    "50v-10.mps",
    "cryptanalysiskb128n5obj16.mps",
    "graphdraw-domain.mps",
    "n3div36.mps",
    "neos-4387871-tavua.mps",
    "neos-960392.mps",
    "rail02.mps",
    "square41.mps",
    "academictimetablesmall.mps",
    "csched007.mps",
    "h80x6320d.mps",
    "n5-3.mps",
    "neos-4413714-turia.mps",
    "net12.mps",
    "rail507.mps",
    "square47.mps",
    "air05.mps",
    "csched008.mps",
    "highschool1-aigio.mps",
    "neos-1122047.mps",
    "neos-4532248-waihi.mps",
    "netdiversion.mps",
    "ran14x18-disj-8.mps",
    "supportcase10.mps",
    "app1-1.mps",
    "cvs16r128-89.mps",
    "hypothyroid-k1.mps",
    "neos-1171448.mps",
    "neos-4647030-tutaki.mps",
    "nexp-150-20-8-5.mps",
    "rd-rplusc-21.mps",
    "supportcase12.mps",
    "app1-2.mps",
    "dano3_3.mps",
    "ic97_potential.mps",
    "neos-1171737.mps",
    "neos-4722843-widden.mps",
    "ns1116954.mps",
    "reblock115.mps",
    "supportcase18.mps",
    "assign1-5-8.mps",
    "dano3_5.mps",
    "icir97_tension.mps",
    "neos-1354092.mps",
    "neos-4738912-atrato.mps",
    "ns1208400.mps",
    "rmatr100-p10.mps",
    "supportcase19.mps",
    "atlanta-ip.mps",
    "decomp2.mps",
    "irish-electricity.mps",
    "neos-1445765.mps",
    "neos-4763324-toguru.mps",
    "ns1644855.mps",
    "rmatr200-p5.mps",
    "supportcase22.mps",
    "b1c1s1.mps",
    "drayage-100-23.mps",
    "irp.mps",
    "neos-1456979.mps",
    "neos-4954672-berkel.mps",
    "ns1760995.mps",
    "rocI-4-11.mps",
    "supportcase26.mps",
    "bab2.mps",
    "drayage-25-23.mps",
    "istanbul-no-cutoff.mps",
    "neos-1582420.mps",
    "neos-5049753-cuanza.mps",
    "ns1830653.mps",
    "rocII-5-11.mps",
    "supportcase33.mps",
    "bab6.mps",
    "dws008-01.mps",
    "k1mushroom.mps",
    "neos17.mps",
    "neos-5052403-cygnet.mps",
    "ns1952667.mps",
    "rococoB10-011000.mps",
    "supportcase40.mps",
    "beasleyC3.mps",
    "eil33-2.mps",
    "lectsched-5-obj.mps",
    "neos-2075418-temuka.mps",
    "neos-5093327-huahum.mps",
    "nu25-pr12.mps",
    "rococoC10-001000.mps",
    "supportcase42.mps",
    "binkar10_1.mps",
    "eilA101-2.mps",
    "leo1.mps",
    "neos-2657525-crna.mps",
    "neos-5104907-jarama.mps",
    "neos-5104907-jarama.mps",
    "nursesched-medium-hint03.mps",
    "roi2alpha3n4.mps",
    "supportcase6.mps",
    "blp-ar98.mps",
    "enlight_hard.mps",
    "leo2.mps",
    "neos-2746589-doon.mps",
    "neos-5107597-kakapo.mps",
    "nursesched-sprint02.mps",
    "roi5alpha10n8.mps",
    "supportcase7.mps",
    "blp-ic98.mps",
    "ex10.mps",
    "lotsize.mps",
    "neos-2978193-inde.mps",
    "neos-5114902-kasavu.mps",
    "nw04.mps",
    "roll3000.mps",
    "swath1.mps",
    "bnatt400.mps",
    "ex9.mps",
    "mad.mps",
    "neos-2987310-joes.mps",
    "neos-5188808-nattai.mps",
    "opm2-z10-s4.mps",
    "s100.mps",
    "swath3.mps",
    "bnatt500.mps",
    "exp-1-500-5-5.mps",
    "map10.mps",
    "neos-3004026-krka.mps",
    "neos-5195221-niemur.mps",
    "p200x1188c.mps",
    "s250r10.mps",
    "tbfp-network.mps",
    "bppc4-08.mps",
    "fast0507.mps",
    "map16715-04.mps",
    "neos-3024952-loue.mps",
    "neos5.mps",
    "peg-solitaire-a3.mps",
    "satellites2-40.mps",
    "thor50dday.mps",
    "brazil3.mps",
    "fastxgemm-n2r6s0t2.mps",
    "markshare2.mps",
    "neos-3046615-murg.mps",
    "neos-631710.mps",
    "pg5_34.mps",
    "satellites2-60-fs.mps",
    "timtab1.mps",
    "buildingenergy.mps",
    "fhnw-binpack4-48.mps",
    "markshare_4_0.mps",
    "neos-3083819-nubu.mps",
    "neos-662469.mps",
    "pg.mps",
    "savsched1.mps",
    "tr12-30.mps",
    "cbs-cta.mps",
    "fhnw-binpack4-4.mps",
    "mas74.mps",
    "neos-3216931-puriri.mps",
    "neos-787933.mps",
    "physiciansched3-3.mps",
    "sct2.mps",
    "traininstance2.mps",
    "chromaticindex1024-7.mps",
    "fiball.mps",
    "mas76.mps",
    "neos-3381206-awhea.mps",
    "neos-827175.mps",
    "physiciansched6-2.mps",
    "seymour1.mps",
    "traininstance6.mps",
    "chromaticindex512-7.mps",
    "gen-ip002.mps",
    "mc11.mps",
    "neos-3402294-bobin.mps",
    "neos-848589.mps",
    "piperout-08.mps",
    "seymour.mps",
    "trento1.mps",
    "cmflsp50-24-8-8.mps",
    "gen-ip054.mps",
    "mcsched.mps",
    "neos-3402454-bohle.mps",
    "neos859080.mps",
    "piperout-27.mps",
    "sing326.mps",
    "triptim1.mps",
    "CMS750_4.mps",
    "germanrr.mps",
    "mik-250-20-75-4.mps",
    "neos-3555904-turama.mps",
    "neos-860300.mps",
    "pk1.mps",
    "sing44.mps",
    "uccase12.mps",
    "co-100.mps",
    "gfd-schedulen180f7d50m30k18.mps",
    "milo-v12-6-r2-40-1.mps",
    "neos-3627168-kasai.mps",
    "neos-873061.mps",
    "proteindesign121hz512p9.mps",
    "snp-02-004-104.mps",
    "uccase9.mps",
    "cod105.mps",
    "glass4.mps",
    "momentum1.mps",
    "neos-3656078-kumeu.mps",
    "neos8.mps",
    "proteindesign122trx11p8.mps",
    "sorrell3.mps",
    "uct-subprob.mps",
    "comp07-2idx.mps",
    "glass-sc.mps",
    "mushroom-best.mps",
    "neos-3754480-nidda.mps",
    "neos-911970.mps",
    "qap10.mps",
    "sp150x300d.mps",
    "unitcal_7.mps",
    "comp21-2idx.mps",
    "gmu-35-40.mps",
    "mzzv11.mps",
    "neos-3988577-wolgan.mps",
    "neos-933966.mps",
    "radiationm18-12-05.mps",
    "sp97ar.mps",
    "var-smallemery-m6j6.mps",
    "cost266-UUE.mps",
    "gmu-35-50.mps",
    "mzzv42z.mps",
    "neos-4300652-rahue.mps",
    "neos-950242.mps",
    "radiationm40-10-02.mps",
    "sp98ar.mps",
    "wachplan.mps",
]

MittelmannInstances = {
    "emps": "https://old.sztaki.hu/~meszaros/public_ftp/lptestset/emps.c",
    "problems": {
        "irish-electricity": [
            "https://plato.asu.edu/ftp/lptestset/irish-electricity.mps.bz2",
            "mps",
        ],
        "physiciansched3-3": [
            "https://plato.asu.edu/ftp/lptestset/physiciansched3-3.mps.bz2",
            "mps",
        ],
        "16_n14": [
            "https://plato.asu.edu/ftp/lptestset/network/16_n14.mps.bz2",
            "mps",
        ],
        "Dual2_5000": [
            "https://plato.asu.edu/ftp/lptestset/Dual2_5000.mps.bz2",
            "mps",
        ],
        "L1_six1000": [
            "https://plato.asu.edu/ftp/lptestset/L1_sixm1000obs.bz2",
            "netlib",
        ],
        "L1_sixm": ["", "mps"],
        "L1_sixm1000obs": [
            "https://plato.asu.edu/ftp/lptestset/L1_sixm1000obs.bz2",
            "netlib",
        ],
        "L1_sixm250": ["", "netlib"],
        "L1_sixm250obs": [
            "https://plato.asu.edu/ftp/lptestset/L1_sixm250obs.bz2",
            "netlib",
        ],
        "L2CTA3D": [
            "https://plato.asu.edu/ftp/lptestset/L2CTA3D.mps.bz2",
            "mps",
        ],
        "Linf_520c": [
            "https://plato.asu.edu/ftp/lptestset/Linf_520c.bz2",
            "netlib",
        ],
        "Primal2_1000": [
            "https://plato.asu.edu/ftp/lptestset/Primal2_1000.mps.bz2",
            "mps",
        ],
        "a2864": ["https://plato.asu.edu/ftp/lptestset/a2864.mps.bz2", "mps"],
        "bdry2": ["https://plato.asu.edu/ftp/lptestset/bdry2.bz2", "netlib"],
        "braun": ["", "mps"],
        "cont1": [
            "https://plato.asu.edu/ftp/lptestset/misc/cont1.bz2",
            "netlib",
        ],
        "cont11": [
            "https://plato.asu.edu/ftp/lptestset/misc/cont11.bz2",
            "netlib",
        ],
        "datt256": [
            "https://plato.asu.edu/ftp/lptestset/datt256_lp.mps.bz2",
            "mps",
        ],
        "datt256_lp": [
            "https://plato.asu.edu/ftp/lptestset/datt256_lp.mps.bz2",
            "mps",
        ],
        "degme": [
            "https://old.sztaki.hu/~meszaros/public_ftp/lptestset/New/degme.gz",
            "netlib",
        ],
        "dlr1": ["https://plato.asu.edu/ftp/lptestset/dlr1.mps.bz2", "mps"],
        "dlr2": ["https://plato.asu.edu/ftp/lptestset/dlr2.mps.bz2", "mps"],
        "energy1": ["", "mps"],  # Kept secret by Mittlemman
        "energy2": ["", "mps"],
        "ex10": ["https://plato.asu.edu/ftp/lptestset/ex10.mps.bz2", "mps"],
        "fhnw-binschedule1": [
            "https://plato.asu.edu/ftp/lptestset/fhnw-binschedule1.mps.bz2",
            "mps",
        ],
        "fome13": [
            "https://plato.asu.edu/ftp/lptestset/fome/fome13.bz2",
            "netlib",
        ],
        "gamora": ["", "mps"],  # Kept secret by Mittlemman
        "goto14_256_1": ["", "mps"],
        "goto14_256_2": ["", "mps"],
        "goto14_256_3": ["", "mps"],
        "goto14_256_4": ["", "mps"],
        "goto14_256_5": ["", "mps"],
        "goto16_64_1": ["", "mps"],
        "goto16_64_2": ["", "mps"],
        "goto16_64_3": ["", "mps"],
        "goto16_64_4": ["", "mps"],
        "goto16_64_5": ["", "mps"],
        "goto32_512_1": ["", "mps"],
        "goto32_512_2": ["", "mps"],
        "goto32_512_3": ["", "mps"],
        "goto32_512_4": ["", "mps"],
        "goto32_512_5": ["", "mps"],
        "graph40-40": [
            "https://plato.asu.edu/ftp/lptestset/graph40-40.mps.bz2",
            "mps",
        ],
        "graph40-40_lp": [
            "https://plato.asu.edu/ftp/lptestset/graph40-40.mps.bz2",
            "mps",
        ],
        "groot": ["", "mps"],  # Kept secret by Mittlemman
        "heimdall": ["", "mps"],  # Kept secret by Mittlemman
        "hulk": ["", "mps"],  # Kept secret by Mittlemman
        "i_n13": [
            "https://plato.asu.edu/ftp/lptestset/network/i_n13.mps.bz2",
            "mps",
        ],
        "irish-e": ["", "mps"],
        "karted": [
            "https://old.sztaki.hu/~meszaros/public_ftp/lptestset/New/karted.gz",
            "netlib",
        ],
        "lo10": [
            "https://plato.asu.edu/ftp/lptestset/network/lo10.mps.bz2",
            "mps",
        ],
        "loki": ["", "mps"],  # Kept secret by Mittlemman
        "long15": [
            "https://plato.asu.edu/ftp/lptestset/network/long15.mps.bz2",
            "mps",
        ],
        "nebula": ["", "mps"],  # Kept secret by Mittlemman
        "neos": [
            "https://plato.asu.edu/ftp/lptestset/misc/neos.bz2",
            "netlib",
        ],
        "neos-3025225": [
            "https://plato.asu.edu/ftp/lptestset/neos-3025225.mps.bz2",
            "mps",
        ],
        "neos-3025225_lp": [
            "https://plato.asu.edu/ftp/lptestset/neos-3025225.mps.bz2",
            "mps",
        ],
        "neos-5251015": [
            "https://plato.asu.edu/ftp/lptestset/neos-5251015.mps.bz2",
            "mps",
        ],
        "neos-5251015_lp": [
            "https://plato.asu.edu/ftp/lptestset/neos-5251015.mps.bz2",
            "mps",
        ],
        "neos3": [
            "https://plato.asu.edu/ftp/lptestset/misc/neos3.bz2",
            "netlib",
        ],
        "neos-5052403-cygnet": [
            "https://plato.asu.edu/ftp/lptestset/neos-5052403-cygnet.mps.bz2",
            "mps",
        ],
        "neos5251015_lp": [
            "https://plato.asu.edu/ftp/lptestset/neos-5251015.mps.bz2",
            "mps",
        ],
        "neos5251915": [
            "https://plato.asu.edu/ftp/lptestset/neos-5251015.mps.bz2",
            "mps",
        ],
        "netlarge1": [
            "https://plato.asu.edu/ftp/lptestset/network/netlarge1.mps.bz2",
            "mps",
        ],
        "netlarge2": [
            "https://plato.asu.edu/ftp/lptestset/network/netlarge2.mps.bz2",
            "mps",
        ],
        "netlarge3": [
            "https://plato.asu.edu/ftp/lptestset/network/netlarge3.mps.bz2",
            "mps",
        ],
        "netlarge6": [
            "https://plato.asu.edu/ftp/lptestset/network/netlarge6.mps.bz2",
            "mps",
        ],
        "ns1687037": [
            "https://plato.asu.edu/ftp/lptestset/misc/ns1687037.bz2",
            "netlib",
        ],
        "ns1688926": [
            "https://plato.asu.edu/ftp/lptestset/misc/ns1688926.bz2",
            "netlib",
        ],
        "nug08-3rd": [
            "https://plato.asu.edu/ftp/lptestset/nug/nug08-3rd.bz2",
            "netlib",
        ],
        "pds-100": [
            "https://plato.asu.edu/ftp/lptestset/pds/pds-100.bz2",
            "netlib",
        ],
        "psched3-3": ["", "mps"],
        "qap15": ["https://plato.asu.edu/ftp/lptestset/qap15.mps.bz2", "mps"],
        "rail02": ["https://miplib2010.zib.de/download/rail02.mps.gz", "mps"],
        "rail4284": [
            "https://plato.asu.edu/ftp/lptestset/rail/rail4284.bz2",
            "netlib",
        ],
        "rmine15": [
            "https://plato.asu.edu/ftp/lptestset/rmine15.mps.bz2",
            "mps",
        ],
        "s100": ["https://plato.asu.edu/ftp/lptestset/s100.mps.bz2", "mps"],
        "s250r10": [
            "https://plato.asu.edu/ftp/lptestset/s250r10.mps.bz2",
            "mps",
        ],
        "s82": ["https://plato.asu.edu/ftp/lptestset/s82.mps.bz2", "mps"],
        "savsched1": [
            "https://plato.asu.edu/ftp/lptestset/savsched1.mps.bz2",
            "mps",
        ],
        "scpm1": ["https://plato.asu.edu/ftp/lptestset/scpm1.mps.bz2", "mps"],
        "set-cover-model": [
            "https://plato.asu.edu/ftp/lptestset/set-cover-model.mps.bz2",
            "mps",
        ],
        "shs1023": [
            "https://miplib2010.zib.de/download/shs1023.mps.gz",
            "mps",
        ],
        "square15": [
            "https://plato.asu.edu/ftp/lptestset/network/square15.mps.bz2",
            "mps",
        ],
        "square41": [
            "https://plato.asu.edu/ftp/lptestset/square41.mps.bz2",
            "mps",
        ],
        "stat96v2": [
            "https://old.sztaki.hu/~meszaros/public_ftp/lptestset/misc/stat96v2.gz",
            "netlib",
        ],
        "stormG2_1000": [
            "https://plato.asu.edu/ftp/lptestset/misc/stormG2_1000.bz2",
            "netlib",
        ],
        "storm_1000": ["", "mps"],
        "stp3d": [
            "https://miplib.zib.de/WebData/instances/stp3d.mps.gz",
            "mps",
        ],
        "supportcase10": [
            "https://plato.asu.edu/ftp/lptestset/supportcase10.mps.bz2",
            "mps",
        ],
        "support19": [
            "https://plato.asu.edu/ftp/lptestset/supportcase19.mps.bz2",
            "mps",
        ],
        "supportcase19": [
            "https://plato.asu.edu/ftp/lptestset/supportcase19.mps.bz2",
            "mps",
        ],
        "test03": ["", "mps"],  # Kept secret by Mittlemman
        "test13": ["", "mps"],  # Kept secret by Mittlemman
        "test23": ["", "mps"],  # Kept secret by Mittlemman
        "test33": ["", "mps"],  # Kept secret by Mittlemman
        "test43": ["", "mps"],  # Kept secret by Mittlemman
        "test53": ["", "mps"],  # Kept secret by Mittlemman
        "test63": ["", "mps"],  # Kept secret by Mittlemman
        "test83": ["", "mps"],  # Kept secret by Mittlemman
        "test93": ["", "mps"],  # Kept secret by Mittlemman
        "mars": ["", "mps"],  # Kept secret by Mittlemman
        "thk_48": [
            "https://plato.asu.edu/ftp/lptestset/thk_48.mps.bz2",
            "mps",
        ],
        "thk_63": [
            "https://plato.asu.edu/ftp/lptestset/thk_63.mps.bz2",
            "mps",
        ],
        "thor": ["", "mps"],  # Kept secret by Mittlemman
        "tpl-tub-ws": ["", "mps"],
        "tpl-tub-ws1617": [
            "https://plato.asu.edu/ftp/lptestset/tpl-tub-ws1617.mps.bz2",
            "mps",
        ],
        "wide15": [
            "https://plato.asu.edu/ftp/lptestset/network/wide15.mps.bz2",
            "mps",
        ],
        "woodlands09": [
            "https://plato.asu.edu/ftp/lptestset/woodlands09.mps.bz2",
            "mps",
        ],
    },
    "benchmarks": {
        "simplex": [
            "L1_sixm",
            "L1_sixm250obs",
            "Linf_520c",
            "a2864",
            "bdry2",
            "braun",
            "cont1",
            "cont11",
            "datt256",
            "dlr1",
            "energy1",
            "energy2",
            "ex10",
            "fhnw-binschedule1",
            "fome13",
            "gamora",
            "graph40-40",
            "groot",
            "heimdall",
            "hulk",
            "irish-e",
            "loki",
            "nebula",
            "neos",
            "neos-3025225_lp",
            "neos-5251015_lp",
            "neos3",
            "neos3025225",
            "neos5052403",
            "neos5251015_lp",
            "ns1687037",
            "ns1688926",
            "nug08-3rd",
            "pds-100",
            "psched3-3",
            "qap15",
            "rail02",
            "rail4284",
            "rmine15",
            "s100",
            "s250r10",
            "s82",
            "savsched1",
            "scpm1",
            "shs1023",
            "square41",
            "stat96v2",
            "stormG2_1000",
            "storm_1000",
            "stp3d",
            "support10",
            "test03",
            "test13",
            "test23",
            "test33",
            "test43",
            "test53",
            "thor",
            "tpl-tub-ws",
            "tpl-tub-ws16",
            "woodlands09",
        ],
        "barrier": [
            "Dual2_5000",
            "L1_six1000",
            "L1_sixm1000obs",
            "L1_sixm250",
            "L1_sixm250obs",
            "L2CTA3D",
            "Linf_520c",
            "Primal2_1000",
            "a2864",
            "bdry2",
            "cont1",
            "cont11",
            "datt256",
            "degme",
            "dlr1",
            "dlr2",
            "ex10",
            "fhnw-binschedule1",
            "fome13",
            "graph40-40",
            "irish-e",
            "karted",
            "neos",
            "neos-3025225_lp",
            "neos-5251015_lp",
            "neos3",
            "neos3025225",
            "neos5052403",
            "neos5251915",
            "ns1687037",
            "ns1688926",
            "nug08-3rd",
            "pds-100",
            "psched3-3",
            "qap15",
            "rail02",
            "rail4284",
            "rmine15",
            "s100",
            "s250r10",
            "s82",
            "savsched1",
            "scpm1",
            "set-cover-model",
            "shs1023",
            "square41",
            "stat96v2",
            "stormG2_1000",
            "storm_1000",
            "stp3d",
            "support10",
            "support19",
            "supportcase19",
            "thk_63",
            "tpl-tub-ws",
            "tpl-tub-ws16",
            "woodlands09",
        ],
        "large": [
            "16_n14",
            "goto14_256_1",
            "goto14_256_2",
            "goto14_256_3",
            "goto14_256_4",
            "goto14_256_5",
            "goto16_64_1",
            "goto16_64_2",
            "goto16_64_3",
            "goto16_64_4",
            "goto16_64_5",
            "goto32_512_1",
            "goto32_512_2",
            "goto32_512_3",
            "goto32_512_4",
            "goto32_512_5",
            "i_n13",
            "lo10",
            "long15",
            "netlarge1",
            "netlarge2",
            "netlarge3",
            "netlarge6",
            "square15",
            "wide15",
        ],
        # <=100s in bench: https://plato.asu.edu/ftp/lpbar.html
        "L0": [
            "ex10",
            "datt256",
            "graph40-40",
            "neos5251915",
            "nug08-3rd",
            "qap15",
            "savsched1",
            "scpm1",
            "a2864",
            "support10",
            "rmine15",
            "fome13",
            "L2CTA3D",
            "neos5052403",
            "karted",
            "stp3d",
            "woodlands09",
            "rail4284",
            "L1_sixm250",
            "tpl-tub-ws",
        ],
        # >100 <1000
        "L1": [
            "s250r10",
            "pds-100",
            "set-cover-model",
            "neos3025225",
            "rail02",
            "square41",
            "degme",
            "Linf_520c",
            "cont1",
            "neos",
            "stat96v2",
            "support19",
            "shs1023",
            "storm_1000",
        ],
        # >1000
        "L2": [
            "thk_63",
            "Primal2_1000",
            "L1_six1000",
            "Dual2_5000",
            "s100",
            "fhnw-binschedule1",
            "cont11",
            "psched3-3",
        ],
        # t -> >15000
        "L3": [
            "dlr2",
            "bdry2",
            "dlr1",
            "irish-e",
            "ns1687037",
            "ns1688926",
            "s82",
        ],
    },
}


def download(url, dst):
    if os.path.exists(dst):
        return
    print(f"Downloading {url} into {dst}...")
    # Bypass SSL verification for plato.asu.edu URLs
    if "plato.asu.edu" in url:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        response = urllib.request.urlopen(url, context=context)
    else:
        response = urllib.request.urlopen(url)
    data = response.read()
    with open(dst, "wb") as fp:
        fp.write(data)


def extract(file, dir, type):
    basefile = os.path.basename(file)
    outfile = ""
    unzippedfile = ""
    if basefile.endswith(".bz2"):
        outfile = basefile.replace(".bz2", ".mps")
        unzippedfile = basefile.replace(".bz2", "")
        subprocess.run(f"cd {dir} && bzip2 -d {basefile}", shell=True)
    elif basefile.endswith(".gz"):
        outfile = basefile.replace(".gz", ".mps")
        unzippedfile = basefile.replace(".gz", "")
        subprocess.run(
            f"cd {dir} && gunzip -c {basefile} > {unzippedfile}", shell=True
        )
        subprocess.run(f"cd {dir} && rm -rf {basefile}", shell=True)
    else:
        raise Exception(f"Unknown file extension found for extraction {file}")
    # download emps and compile
    # Disable emps for now
    if type == "netlib":
        url = MittelmannInstances["emps"]
        file = os.path.join(dir, "emps.c")
        download(url, file)
        subprocess.run(
            f"cd {dir} && gcc -Wno-implicit-int emps.c -o emps", shell=True
        )
        # determine output file and run emps
        subprocess.run(
            f"cd {dir} && ./emps {unzippedfile} > {outfile}", shell=True
        )
        subprocess.run(f"cd {dir} && rm -rf {unzippedfile}", shell=True)
        # cleanup emps and emps.c
        subprocess.run(f"rm -rf {dir}/emps*", shell=True)


def download_lp_dataset(name, dir):
    if name not in MittelmannInstances["problems"]:
        raise Exception(f"Unknown dataset {name} passed")
    if os.path.exists(dir):
        if os.path.exists(os.path.join(dir, f"{name}.mps")):
            print(
                f"Dir for dataset {name} exists and contains {name}.mps. Skipping..."
            )
            return
    url, type = MittelmannInstances["problems"][name]
    if url == "":
        print(f"Dataset {name} doesn't have a URL. Skipping...")
        return
    file = os.path.join(dir, os.path.basename(url))
    download(url, file)
    extract(file, dir, type)


def download_mip_dataset(name, dir):
    base_url = "https://miplib.zib.de/WebData/instances"
    url = f"{base_url}/{name}.gz"
    outfile = f"{dir}/{name}.gz"
    if os.path.exists(dir):
        if os.path.exists(os.path.join(dir, f"{name}")):
            print(
                f"Dir for dataset {name} exists and contains {name}.mps. Skipping..."
            )
            return
    download(url, outfile)
    extract(outfile, dir, "")


datasets_path = sys.argv[1]
dataset_type = sys.argv[2]

if dataset_type == "lp":
    for name in LPFeasibleMittelmannSet:
        download_lp_dataset(name, datasets_path)
elif dataset_type == "mip":
    for name in MiplibInstances:
        download_mip_dataset(name, datasets_path)
