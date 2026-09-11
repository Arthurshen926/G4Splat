from copy import deepcopy
import pytest
from scripts.compare_native_canopy_audits import compare


def fixture():
    return dict(scope='test',leaf_optical_kernel='native',experimental_binary_sha256=None,
        audit_script_sha256='a',rgb_region_helper_sha256='b',checkpoint_sha256='c',
        records=[dict(index=1,image_shape=[1080,1920],canopy_surface_alpha_after=.3,
            rgb_regions=dict(source=dict(tree=15.),updated=dict(tree=16.)))])


def test_matched_difference():
    a=fixture();b=deepcopy(a);b['records'][0]['rgb_regions']['updated']['tree']=17.
    assert compare(a,b)['mean_delta']==dict(tree=1.)


@pytest.mark.parametrize('change',['resolution','source','duplicate'])
def test_mismatched_audit_rejected(change):
    a=fixture();b=deepcopy(a)
    if change=='resolution':b['records'][0]['image_shape']=[360,640]
    if change=='source':b['records'][0]['rgb_regions']['source']['tree']=14.
    if change=='duplicate':b['records']*=2
    with pytest.raises(ValueError):compare(a,b)
