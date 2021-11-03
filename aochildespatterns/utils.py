import pandas as pd
import numpy as np
from typing import List

from aochildespatterns import configs


def save_summary_to_txt(x: List[int],
                        y: np.array,
                        quantity_name: str,
                        ) -> None:
    """
    output summary to text file.
    useful when plotting data with pgfplots on overleaf.org.

    notes:
        1. latex does not like spaces in file name
    """

    # make path
    fn = quantity_name + '_in_aochildes'
    path = configs.Dirs.summaries / f'{fn}.txt'  # txt format makes file content visible on overleaf.org
    if not path.parent.exists():
        path.parent.mkdir()

    # save to text
    df = pd.DataFrame(data={'mean': y}, index=x)
    df.index.name = 'x'
    df.round(3).to_csv(path, sep=' ')

    print(f'Saved summary to {path}')
