"""Testing automatic BIDS report."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD-3-Clause
import os.path as op
import textwrap

import mne
import pytest
from mne.datasets import testing

from mne_bids import BIDSPath, make_report
from mne_bids.write import write_raw_bids
from mne_bids.config import BIDS_VERSION

from mne_bids.report._report import (
    _range_str,
    _summarize_participant_hand,
    _summarize_participant_sex,
)

subject_id = "01"
session_id = "01"
run = "01"
acq = "01"
task = "testing"

_bids_path = BIDSPath(
    subject=subject_id, session=session_id, run=run, acquisition=acq, task=task
)

# Get the MNE testing sample data
data_path = testing.data_path(download=False)
raw_fname = op.join(data_path, "MEG", "sample", "sample_audvis_trunc_raw.fif")

warning_str = dict(
    channel_unit_changed="ignore:The unit for chann*.:RuntimeWarning:mne",
)

def test_summarize_participant_sex():
    # Test case 1: All sexes are unknown
    result = _summarize_participant_sex(["n/a", "n/a", "n/a"])
    assert result == "sex were all unknown"

    # Test case 2: Equal number of males and females
    result = _summarize_participant_sex(["M", "F", "M", "F"])
    assert result == "comprised of 2 male and 2 female participants"

    # Test case 3: Only males
    result = _summarize_participant_sex(["M", "M", "M"])
    assert result == "comprised of 3 male and 0 female participants"

    # Test case 4: Only females
    result = _summarize_participant_sex(["F", "F"])
    assert result == "comprised of 0 male and 2 female participants"

    # Test case 5: Mix of sexes
    result = _summarize_participant_sex(["M", "F", "n/a", "M", "n/a", "F"])
    assert result == "comprised of 2 male and 2 female participants"

    # Test case 6: Empty list
    result = _summarize_participant_sex([])
    assert result == "sex were all unknown"

def test_range_str():
    # Test case 1: All values are valid
    result = _range_str(10, 20, 15, 2, 0, "years")
    expected_result = "ages ranged from 10 to 20 (mean = 15, std = 2)"
    assert result == expected_result

    # Test case 2: Minval is "n/a"
    result = _range_str("n/a", 20, 15, 2, 0, "years")
    expected_result = "ages all unknown"
    assert result == expected_result

    # Test case 3: n_unknown is greater than 0
    result = _range_str(10, 20, 15, 2, 5, "years")
    expected_result = "ages ranged from 10 to 20 (mean = 15, std = 2; 5 with unknown years)"
    assert result == expected_result

def test_summarize_participant_hand():
    # Test case 1: All hands are unknown
    hands_case1 = ["n/a", "n/a", "n/a"]
    result_case1 = _summarize_participant_hand(hands_case1)
    assert result_case1 == "handedness were all unknown"

    # Test case 2: Mixed hands
    hands_case2 = ["R", "L", "A", "R", "A"]
    result_case2 = _summarize_participant_hand(hands_case2)
    expected_result_case2 = "comprised of 2 right hand, 1 left hand and 2 ambidextrous"
    assert result_case2 == expected_result_case2

    # Test case 3: All left hands
    hands_case3 = ["L", "L", "L"]
    result_case3 = _summarize_participant_hand(hands_case3)
    expected_result_case3 = "comprised of 0 right hand, 3 left hand and 0 ambidextrous"
    assert result_case3 == expected_result_case3

@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_report(tmp_path):
    """Test that report generated works as intended."""
    bids_root = str(tmp_path)
    raw = mne.io.read_raw_fif(raw_fname, verbose=False)
    raw.info["line_freq"] = 60
    bids_path = _bids_path.copy().update(root=bids_root)
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    report = make_report(bids_root)

    expected_report = f"""This dataset was created by [Unspecified] and conforms to BIDS version {BIDS_VERSION}.
This report was generated with MNE-BIDS (https://doi.org/10.21105/joss.01896).
The dataset consists of 1 participants (sex were all unknown; handedness were
all unknown; ages all unknown) and 1 recording sessions: 01. Data was recorded
using an MEG system (Elekta) sampled at 300.31 Hz with line noise at
60.0 Hz. The following software filters were applied during recording:
SpatialCompensation. There was 1 scan in total. Recording durations ranged from
20.0 to 20.0 seconds (mean = 20.0, std = 0.0), for a total of 20.0 seconds of
data recorded over all scans. For each dataset, there were on average 376.0 (std
= 0.0) recording channels per scan, out of which 374.0 (std = 0.0) were used in
analysis (2.0 +/- 0.0 were removed from analysis)."""  # noqa

    expected_report = "\n".join(textwrap.wrap(expected_report, width=80))
    assert report == expected_report


@pytest.mark.filterwarnings(warning_str["channel_unit_changed"])
@testing.requires_testing_data
def test_report_no_participant_information(tmp_path):
    """Test report with participants.tsv with participant_id column only."""
    bids_root = tmp_path
    raw = mne.io.read_raw_fif(raw_fname, verbose=False)
    raw.info["line_freq"] = 60
    bids_path = _bids_path.copy().update(root=bids_root)
    write_raw_bids(raw, bids_path, overwrite=True, verbose=False)

    # remove all information and check if report still runs
    (bids_root / "participants.json").unlink()

    # overwrite participant information to see if report still runs
    (bids_root / "participants.tsv").write_text("participant_id\nsub-001")

    report = make_report(bids_root)

    expected_report = f"""This dataset was created by [Unspecified] and conforms to BIDS version {BIDS_VERSION}.
This report was generated with MNE-BIDS (https://doi.org/10.21105/joss.01896).
The dataset consists of 1 participants (sex were all unknown; handedness were
all unknown; ages all unknown) and 1 recording sessions: 01. Data was recorded
using an MEG system (Elekta) sampled at 300.31 Hz with line noise at
60.0 Hz. The following software filters were applied during recording:
SpatialCompensation. There was 1 scan in total. Recording durations ranged from
20.0 to 20.0 seconds (mean = 20.0, std = 0.0), for a total of 20.0 seconds of
data recorded over all scans. For each dataset, there were on average 376.0 (std
= 0.0) recording channels per scan, out of which 374.0 (std = 0.0) were used in
analysis (2.0 +/- 0.0 were removed from analysis)."""  # noqa

    expected_report = "\n".join(textwrap.wrap(expected_report, width=80))
    assert report == expected_report
