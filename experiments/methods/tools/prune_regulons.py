import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ctxcore.genesig import Regulon
from ctxcore.rnkdb import FeatherRankingDatabase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("rankings", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("workers", type=int)
    args = parser.parse_args()
    # pySCENIC uses this removed alias only in its metadata dtype declaration.
    np.object = object
    try:
        from pyscenic.prune import prune2df
    finally:
        del np.object
    table = pd.read_pickle(args.work / "modules.pkl")
    modules = [
        Regulon(
            name=name,
            gene2weight=dict(zip(group.target, group.weight)),
            gene2occurrence={},
            transcription_factor=group.TF.iloc[0],
            context=frozenset((group.direction.iloc[0], group.context.iloc[0])),
        )
        for name, group in table.groupby("source", sort=True)
    ]
    result = prune2df(
        [FeatherRankingDatabase(str(args.rankings), "hg38")],
        modules,
        str(args.annotations),
        rank_threshold=5000,
        auc_threshold=0.05,
        nes_threshold=3.0,
        weighted_recovery=False,
        client_or_address="custom_multiprocessing",
        num_workers=args.workers,
        module_chunksize=20,
    )
    result.to_pickle(args.work / "reference_motifs.pkl")
    result.to_csv(args.work / "reference_motifs.csv.gz", compression="gzip")


if __name__ == "__main__":
    main()
