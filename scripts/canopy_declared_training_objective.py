"""One definition of the diagnostic RGB objective for training and step replay."""
from scripts.canopy_candidate_rgb_ownership import candidate_rgb_loss
from scripts.canopy_candidate_ray_refinement import rigid_rgb_preservation
from scripts.canopy_boundary_preservation import boundary_rgb_preservation
from scripts.canopy_surface_rgb_feasibility import surface_rgb_feasibility_loss
from scripts.canopy_relative_radiance_feasibility import relative_surface_rgb_feasibility_loss


def declared_objective(rgb,target,reference,regions,args,*,floor_rgb=None,relative_scale=1.,
                       surface_alpha=None,reference_surface_alpha=None):
    observed=args.noncanopy_rgb_target=='observed_risk'
    if observed:
        from scripts.canopy_observed_background_risk import observed_rgb_risk
    loss=candidate_rgb_loss(rgb,target,reference,regions,noncanopy_target=args.noncanopy_rgb_target)
    if args.rigid_rgb_preservation_weight:
        guard=(observed_rgb_risk(rgb,target,reference,regions['rigid']) if observed
               else rigid_rgb_preservation(rgb,reference,regions['rigid']))
        loss=loss+args.rigid_rgb_preservation_weight*guard
    boundary=rgb.new_zeros(());floor=rgb.new_zeros(())
    if args.rigid_boundary_preservation_weight:
        boundary=(observed_rgb_risk(rgb,target,reference,regions['rigid']&regions['hard']) if observed
                  else boundary_rgb_preservation(rgb,reference,regions['rigid'],regions['hard']))
        loss=loss+args.rigid_boundary_preservation_weight*boundary
    if args.surface_rgb_feasibility_weight:
        if floor_rgb is None:raise ValueError('Actual native surface-radiance render required')
        floor=(relative_surface_rgb_feasibility_loss(floor_rgb,target,regions['tree'],relative_scale)
               if args.surface_rgb_feasibility_domain=='relative' else surface_rgb_feasibility_loss(floor_rgb,target,regions['tree']))
        loss=loss+args.surface_rgb_feasibility_weight*floor
    visibility_weight=getattr(args,'visible_rigid_alpha_weight',0.)
    if visibility_weight:
        if surface_alpha is None or reference_surface_alpha is None:
            raise ValueError('Actual current and immutable surface contributions required')
        from scripts.canopy_observed_background_risk import visible_rigid_loss
        loss=loss+visibility_weight*visible_rigid_loss(surface_alpha,reference_surface_alpha,reference,target,regions)
    return loss,boundary,floor
