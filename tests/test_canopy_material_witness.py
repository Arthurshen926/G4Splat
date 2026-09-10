from types import SimpleNamespace
import pytest
import torch
from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel
from outdoor.canopy_support_expansion import canopy_material_audit_fields
from scripts.train_unified_outdoor_teacher import _update_static_child_verification_from_render


def child():
    foliage=VolumetricFoliageModel(1,device='cpu')
    foliage.initialize_from_volume_state({'version':'independent_sfm_semantic_canopy_volume_v1',
        'centers':torch.tensor([[0.,0.,2.]]),'scales':torch.full((1,3),.1),
        'colors':torch.full((1,3),.9),'opacities':torch.full((1,1),.2),
        'quaternions':torch.tensor([[1.,0.,0.,0.]]),'layer_role':torch.tensor([0],dtype=torch.int8),
        'support_camera_ids':torch.tensor([[7,8]],dtype=torch.int32),
        'support_sequence_count':torch.tensor([1],dtype=torch.int16),
        'evidence_primitive_id':torch.tensor([11],dtype=torch.int64)})
    foliage.split(torch.tensor([0]),birth_iteration=20)
    return foliage


def test_material_audit_rgb_has_no_sky_or_wall_color_and_fits_native_budget():
    canopy=torch.tensor([[1.,0.,0.]])
    rgb=torch.tensor([[[.1,1.,.8]],[[.2,1.,.8]],[[.3,1.,.8]]])
    fields=canopy_material_audit_fields(canopy,torch.tensor([[0.,0.,1.]]),torch.tensor([[0.,1.,0.]]),rgb)
    assert fields.shape==(7,1,3)
    assert torch.equal(fields[3:6,:,1:],torch.zeros(3,1,2))


@pytest.mark.parametrize('masked,update',[(True,True),(False,True),(True,False)])
def test_leaf_color_is_canopy_only_or_preserved_never_whole_footprint(masked,update):
    foliage=child();before=foliage.features.detach().clone()
    responsibility=torch.zeros(len(foliage),8);responsibility[0,0]=1.;responsibility[0,1]=.8;responsibility[0,7]=.2
    dark=torch.tensor([.1,.2,.3]);responsibility[0,4:7]=.8*dark+(0. if masked else .2)
    audit=_update_static_child_verification_from_render(foliage,
        SimpleNamespace(structural_count=0,responsibility=responsibility),camera_id=7,
        camera_sequence_lookup=torch.zeros(9,dtype=torch.int16),required_sequence_count=1,
        canopy_color_audit=masked,sky_audit_column=7,update_color=update)
    assert audit['new_camera_witnesses']==1
    if masked and update:
        torch.testing.assert_close(foliage.features[0,0]*.28209479177387814+.5,dark)
    else:
        assert audit['new_color_witnesses']==0 and torch.equal(foliage.features,before)


def test_sky_dominated_footprint_cannot_become_canopy_witness():
    foliage=child();before=foliage.features.detach().clone()
    responsibility=torch.zeros(len(foliage),8);responsibility[0,0]=1.;responsibility[0,1]=.1;responsibility[0,7]=.9
    audit=_update_static_child_verification_from_render(foliage,
        SimpleNamespace(structural_count=0,responsibility=responsibility),camera_id=7,
        camera_sequence_lookup=torch.zeros(9,dtype=torch.int16),required_sequence_count=1,
        canopy_color_audit=True,sky_audit_column=7)
    assert audit['new_camera_witnesses']==0 and torch.equal(foliage.features,before)


def test_tiny_canopy_mass_cannot_borrow_large_unknown_footprint_as_a_witness():
    foliage=child()
    responsibility=torch.zeros(len(foliage),8);responsibility[0,0]=1.;responsibility[0,1]=1.e-9
    audit=_update_static_child_verification_from_render(foliage,
        SimpleNamespace(structural_count=0,responsibility=responsibility),camera_id=7,
        camera_sequence_lookup=torch.zeros(9,dtype=torch.int16),required_sequence_count=1,
        canopy_color_audit=True,sky_audit_column=7)
    assert audit['new_camera_witnesses']==0
