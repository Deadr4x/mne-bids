from pathlib import Path
import os
import os.path as op
import openneuro

from mne.datasets import sample
from mne_bids import (
    BIDSPath,
    read_raw_bids,
    print_dir_tree,
    make_report,
    find_matching_paths,
    get_entity_vals,
)
import mne_bids
from mne_bids import BIDSPath, get_entity_vals

dataset = "ds003848"
subject = "RESP0521"

# Download one subject's data from each dataset
bids_root = op.join(op.dirname(sample.data_path()), dataset)
if not op.isdir(bids_root):
    os.makedirs(bids_root)


root = Path('C:/Users/Tommy/PycharmProjects/mne-bids/mne_bids/tests/data/tiny_bids').absolute()
entity_key = 'recording'
test = get_entity_vals(bids_root, entity_key)
print(test)

