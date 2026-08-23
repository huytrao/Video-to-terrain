"""Derive the reconstruction knobs from the scene, instead of inheriting them.

`paper_runner.py` carries four constants copied from the paper's `mesh.sh`:

    RENDER_MAX_DEPTH = "4.0"
    MASK_MAX_DEPTH   = "6.0"
    VOXEL_SIZE       = "0.006"
    NUM_CLUSTER      = "1"

Every one is a length or a count that was tuned on ScanNet++ bedrooms, and three
of the four silently destroy geometry when the scene is not a bedroom. They are
therefore both a quality problem and a general-method problem, and the same fix
serves both: express each as a ratio of something this scene supplies, then
measure it per scene.

What each constant actually does, read from the code rather than the docs:

`max_depth` -- `PGSR/render.py:122` does `ref_depth[ref_depth > max_depth] = 0`
    and `render.py:131` passes it again as Open3D's `depth_trunc`. Depth further
    than this from the camera is not "downweighted"; it is erased before fusion.
    At 4.0 units, any wall more than 4 units from every camera is simply absent
    from the mesh. A garage or factory bay is routinely wider than that.

`voxel_size` -- `render.py:150-151`, the TSDF voxel edge and `sdf_trunc =
    4 * voxel_size`. This is the mesh's resolution. Held fixed while the scene
    grows, it means a large room is reconstructed at the same absolute detail as
    a small one, so relative detail falls; held fixed while the scene shrinks, it
    wastes memory. What should be fixed is the ratio of scene size to voxel size.

`num_cluster` -- `render.py:166` `post_process_mesh(mesh, num_cluster)` keeps the
    N largest connected components. At N=1 the mesh is reduced to a single
    component. A room shell is usually that component; a freestanding machine,
    tool chest, or crate that the reconstruction did not weld to the floor is
    not, and is deleted. That directly costs the semantic-boxes and
    collision-mesh deliverables the objects they are about.

`use_depth_filter` -- `render.py:105`, default off. With it off the fused surface
    carries multi-view-inconsistent noise; this project measured 140,334
    connected components on one scene. Note the interaction, which is the part
    that matters: `num_cluster=1` is only survivable *because* the filter is off.
    The cluster cull is doing the filter's job bluntly, and paying for it by
    discarding real disconnected objects along with the noise. Turning the filter
    on is what makes a larger `num_cluster` safe.

So this module takes the depth DA3 actually predicted and the camera path it
actually recovered, and returns knobs with the reasoning attached — including,
for each one, what the inherited constant would have done to *this* scene. That
comparison is the control: a derived number with nothing to compare against is
just a different guess.

Units: when scale mode is `assumed`, these lengths are in DA3's arbitrary units,
not metres. That is exactly why a hardcoded `4.0` cannot be right — there is no
fixed relationship between one unit and one metre from one video to the next.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np


@dataclass
class KnobPolicy:
    """The ratios. These are the only free parameters, and none is a length."""

    # Fuse depth out to the percentile of observed depth that keeps the room.
    # Not 100: the tail of a monocular depth prediction is its least reliable
    # part, and a few stray far samples would drag the truncation out and pull
    # noise in with it.
    depth_percentile: float = 98.0
    # Masking tolerates more range than fusion does -- it is comparing two
    # renders, not integrating a surface -- so it gets a multiple of the fusion
    # distance, mirroring the paper's own 6.0-against-4.0 ratio of 1.5.
    mask_depth_multiple: float = 1.5
    # --- voxel size: the mesh's resolution, and the top priority for this run ---
    #
    # The right reference for "how fine should the mesh be" is not the room and
    # not a remembered constant: it is the resolution of the input. One pixel at
    # depth d covers d / f of the world, where f is the focal length in pixels.
    # A voxel much finer than that reconstructs sensor noise; a voxel much coarser
    # throws away detail the images actually contain. So the voxel is set to a
    # small multiple of the pixel footprint at the depth where most surface sits.
    #
    # This is scale-free by construction -- it never mentions a metre -- so it is
    # also the general-method answer, and it sharpens automatically when the input
    # is higher resolution or the camera was closer.
    pixels_per_voxel: float = 1.75
    # Both rails on the voxel are expressed as a count of voxels across the
    # scene, never as a length. A length would be a hardcoded metre in units that
    # have no fixed relationship to a metre, and the two rails would be free to
    # contradict each other -- an absolute coarse cap can and did override the
    # fine-side memory guard, asking for a 47,000-voxel grid on a large scene.
    # Stated in the same quantity, they cannot cross as long as max > min.
    max_voxels_across_diagonal: float = 6000.0   # memory: no finer than this
    min_voxels_across_diagonal: float = 200.0    # usefulness: no coarser than this
    # Keep every mesh component at least this fraction of the largest one, so
    # freestanding objects survive while specks do not. Applied after fusion,
    # where component sizes are knowable.
    keep_cluster_frac: float = 0.002
    max_clusters: int = 200
    # The multi-view consistency filter, on. See the module docstring for why
    # this and the cluster count have to move together.
    use_depth_filter: bool = True

    # --- PGSR view pairing and schedule (see `pgsr_schedule`) ---
    # Pair a frame with neighbours no closer than this fraction of the typical
    # inter-camera step, so a duplicate viewpoint contributes no constraint while
    # genuinely adjacent frames still do.
    pair_min_step_fraction: float = 0.5
    # ...and no further than this fraction of the scene's extent, beyond which two
    # views stop overlapping and "consistency" between them is meaningless.
    pair_max_extent_fraction: float = 0.25
    # The setting the paper's schedule was tuned in: 30,000 iterations over 261
    # images. Used only as a ratio, never as an absolute count.
    reference_images: int = 261
    reference_iterations: int = 30_000
    # --- densification: what actually bounds the final gaussian count ---
    #
    # PGSR densifies wherever the view-space positional gradient exceeds
    # `densify_grad_threshold` (0.0002) — a threshold tuned for object-scale captures,
    # not rooms. The measured consequence: 1,247,783 gaussians for the pilot's
    # ~70 m^2 room and only 636,362 for this larger factory bay, i.e. *fewer*
    # primitives spread over *more* surface. Since the mesh is fused from
    # PGSR-rendered depth, gaussian density is the binding constraint on surface
    # quality once depth resolution is dealt with.
    #
    # Expressed as a multiplier of the paper's value rather than an absolute, because
    # the gradient's scale depends on the loss and the image resolution and a bare
    # number would not transfer. Halving roughly doubles the primitive count; the
    # cost is memory and time, both of which this GPU has.
    densify_grad_multiple: float = 0.5
    paper_densify_grad_threshold: float = 0.0002
    paper_densify_abs_grad_threshold: float = 0.0008


@dataclass
class Knobs:
    render_max_depth: float
    mask_max_depth: float
    voxel_size: float
    num_cluster: int
    use_depth_filter: bool
    scale_mode: str
    evidence: dict = field(default_factory=dict)

    def as_cli(self) -> dict[str, str]:
        """Stage arguments, as strings, in the form `paper_runner` passes them."""
        return {
            "render_max_depth": f"{self.render_max_depth:.4f}",
            "mask_max_depth": f"{self.mask_max_depth:.4f}",
            "voxel_size": f"{self.voxel_size:.6f}",
            "num_cluster": str(self.num_cluster),
        }


def depth_statistics(depth: np.ndarray, conf: np.ndarray | None = None,
                     conf_percentile: float = 20.0) -> dict:
    """Summarise the depth DA3 predicted, over the samples worth believing.

    Invalid and non-positive depth is excluded. Where confidence is available the
    least-confident tail is excluded too, because the far tail of the depth
    distribution and the low-confidence tail are largely the same pixels, and
    letting them set the fusion distance is how noise gets invited in.
    """
    d = np.asarray(depth, dtype=np.float32).reshape(-1)
    valid = np.isfinite(d) & (d > 0)
    if conf is not None:
        c = np.asarray(conf, dtype=np.float32).reshape(-1)
        if c.shape == d.shape:
            thresh = np.percentile(c[valid], conf_percentile) if valid.any() else 0.0
            valid &= c >= thresh
    d = d[valid]
    if d.size == 0:
        raise ValueError("no valid depth samples; cannot derive fusion distance")

    qs = [50, 75, 90, 95, 98, 99, 99.9]
    return {
        "samples": int(d.size),
        "valid_fraction": float(valid.mean()),
        "min": float(d.min()),
        "max": float(d.max()),
        "mean": float(d.mean()),
        "percentiles": {str(q): float(np.percentile(d, q)) for q in qs},
    }


def derive(depth: np.ndarray, w2c: np.ndarray, K: np.ndarray, conf: np.ndarray | None = None,
           policy: KnobPolicy | None = None, scale_mode: str = "assumed") -> Knobs:
    """Turn measured depth, a recovered camera path, and the camera into stage args.

    `K` must be at the same resolution as `depth`, because the whole point of the
    voxel derivation is the world-space size of one of *these* pixels.
    """
    policy = policy or KnobPolicy()
    stats = depth_statistics(depth, conf)

    # Scene extent has two independent estimates, and they answer different
    # questions: where the cameras went, and how far they could see. A fusion
    # distance must cover the second; the memory backstop needs the union, since
    # that is what the mesh has to span.
    centres = np.stack([np.linalg.inv(_to4(m))[:3, 3] for m in w2c])
    cam_extent = float(np.linalg.norm(centres.max(axis=0) - centres.min(axis=0)))

    fuse = float(stats["percentiles"][str(policy.depth_percentile)]
                 if str(policy.depth_percentile) in stats["percentiles"]
                 else np.percentile(np.asarray(depth).reshape(-1), policy.depth_percentile))
    mask = fuse * policy.mask_depth_multiple
    scene_diagonal = cam_extent + 2.0 * fuse

    K = np.asarray(K, dtype=np.float64)
    f = float(np.median([K[..., 0, 0], K[..., 1, 1]]))
    if not np.isfinite(f) or f <= 0:
        raise ValueError(f"focal length {f} is not usable; cannot size a voxel")
    # The depth most surface sits at, which is what the voxel should resolve --
    # not the far tail, which is where the prediction is least trustworthy.
    d_typical = float(stats["percentiles"]["50"])
    pixel_footprint = d_typical / f
    voxel_ideal = policy.pixels_per_voxel * pixel_footprint

    # Two rails, both counted in voxels across the scene, so they cannot cross.
    # The fine side is a memory guard; the coarse side keeps the mesh worth
    # having. Which one bound the result is recorded, because "the voxel is
    # 0.004" is not a finding but "the input's resolution set it" is.
    if policy.max_voxels_across_diagonal <= policy.min_voxels_across_diagonal:
        raise ValueError(
            f"max_voxels_across_diagonal ({policy.max_voxels_across_diagonal}) must "
            f"exceed min ({policy.min_voxels_across_diagonal})"
        )
    voxel_fine_limit = scene_diagonal / policy.max_voxels_across_diagonal
    voxel_coarse_limit = scene_diagonal / policy.min_voxels_across_diagonal
    voxel = float(np.clip(voxel_ideal, voxel_fine_limit, voxel_coarse_limit))
    if voxel == voxel_ideal:
        bound_by = "input resolution"
    elif voxel == voxel_fine_limit:
        bound_by = "memory guard (max_voxels_across_diagonal)"
    else:
        bound_by = "usefulness cap (min_voxels_across_diagonal)"

    # Components are only countable once a mesh exists, so fusion is asked to
    # keep many and `prune_components` does the size-relative cull afterwards on
    # real numbers. Keeping N here is not the same as keeping noise: the depth
    # filter is on, which is what makes this safe.
    num_cluster = policy.max_clusters

    d = np.asarray(depth, dtype=np.float32).reshape(-1)
    finite = d[np.isfinite(d) & (d > 0)]
    paper_loss = float((finite > 4.0).mean()) if finite.size else float("nan")
    derived_loss = float((finite > fuse).mean()) if finite.size else float("nan")

    evidence = {
        "depth": stats,
        "camera_path_extent": cam_extent,
        "scene_diagonal_estimate": scene_diagonal,
        "policy": asdict(policy),
        "voxel_reasoning": {
            "focal_px": f,
            "typical_depth": d_typical,
            "pixel_footprint_at_typical_depth": pixel_footprint,
            "voxel_at_input_resolution": voxel_ideal,
            "voxel_fine_limit": voxel_fine_limit,
            "voxel_coarse_limit": voxel_coarse_limit,
            "bound_by": bound_by,
            "voxels_across_diagonal": scene_diagonal / voxel,
            # How many pixels wide the paper's constant would be on this camera.
            # If this is much above pixels_per_voxel, the paper was leaving mesh
            # detail on the table for this input; much below, and it was fusing
            # noise.
            "paper_0.006_in_pixels": 0.006 * f / d_typical,
        },
        # The control. Without this the derived numbers are unfalsifiable.
        "against_the_inherited_constants": {
            "paper_render_max_depth": 4.0,
            "fraction_of_valid_depth_erased_at_4.0": paper_loss,
            "fraction_of_valid_depth_erased_at_derived": derived_loss,
            "paper_voxel_size": 0.006,
            "derived_over_paper_voxel": voxel / 0.006,
            "paper_num_cluster": 1,
            "note": (
                "the 4.0 figure is the fraction of believable depth samples the "
                "paper's constant would delete on this scene; if it is near zero "
                "the constant was harmless here and the derived value should be "
                "close to it"
            ),
        },
        "scale_warning": (
            "these are DA3 units, not metres" if scale_mode == "assumed" else None
        ),
    }

    return Knobs(render_max_depth=fuse, mask_max_depth=mask, voxel_size=voxel,
                 num_cluster=num_cluster, use_depth_filter=policy.use_depth_filter,
                 scale_mode=scale_mode, evidence=evidence)


def pgsr_schedule(n_images: int, w2c: np.ndarray, policy: KnobPolicy | None = None) -> dict:
    """Derive PGSR's schedule and view-pairing distances for this scene.

    Two families of PGSR default carry a hidden assumption about the input.

    *View-pairing distances.* `multi_view_min_dis = 0.01` and
    `multi_view_max_dis = 1.5` (`arguments/__init__.py:61-62`) are **lengths in
    scene units**. They decide which neighbouring views a frame is asked to be
    geometrically consistent with -- and multi-view consistency is the mechanism
    the paper credits for a clean surface (section 3.1), so getting them wrong
    degrades the mesh specifically. Too small a maximum and no pair qualifies, so
    the regulariser silently does nothing; too large and pairs share no overlap,
    so it enforces consistency between views that see different things.

    The scene supplies both: the minimum should sit below the typical distance
    between consecutive cameras (so adjacent frames pair) and above zero (so
    duplicate viewpoints do not), and the maximum should be a fraction of the
    scene's extent (so pairs still overlap).

    *Iteration count.* PGSR samples one image per iteration, so 30,000 iterations
    is 115 views of each image at 261 images and only 37 at 799. Holding the
    schedule fixed while the image count grows quietly reduces per-image
    supervision, which is the opposite of what more input should buy. Every
    milestone scales with it, so densification and the two regularisers keep the
    same share of the run.

    *The learning-rate horizon.* `position_lr_max_steps = 30_000`
    (`arguments/__init__.py:83`) is the length of the position learning-rate
    decay, and it is a *separate* number from `iterations`. Raising the iteration
    count alone leaves the schedule decaying to its floor at 30,000 and then
    training on at a position LR that can no longer move a gaussian — so the extra
    time refines colour and opacity while the geometry is frozen. It has to scale
    with the run or the longer run buys much less than it appears to.

    `test_iterations` and `save_iterations` are returned for the same reason,
    though only the first matters: `train.py:532` already does
    `args.save_iterations.append(args.iterations)`, so the final gaussians are
    saved whatever is passed, but `test_iterations` defaults to `[7000, 30000]`
    and would report no PSNR at all at the end of a longer run.
    """
    policy = policy or KnobPolicy()
    centres = np.stack([np.linalg.inv(_to4(m))[:3, 3] for m in w2c])
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    step_median = float(np.median(steps)) if len(steps) else 0.0
    extent = float(np.linalg.norm(centres.max(axis=0) - centres.min(axis=0)))

    min_dis = policy.pair_min_step_fraction * step_median
    max_dis = policy.pair_max_extent_fraction * extent

    scale = n_images / policy.reference_images
    iterations = int(round(policy.reference_iterations * scale / 1000.0) * 1000)
    # A schedule shorter than the reference is still a schedule; a schedule of
    # zero is not. Every milestone below is a fraction of `iterations`, so this
    # floor is what keeps densify_from_iter < densify_until_iter < iterations on a
    # short clip rather than collapsing them all onto 0.
    iterations = max(iterations, 1000)

    def at(frac_of_reference: int) -> int:
        return int(round(frac_of_reference * scale / 500.0) * 500)

    return {
        "iterations": iterations,
        "densify_from_iter": at(500),
        "densify_until_iter": at(15_000),
        "single_view_weight_from_iter": at(7_000),
        "multi_view_weight_from_iter": at(7_000),
        "multi_view_min_dis": min_dis,
        "multi_view_max_dis": max_dis,
        # Tied to `iterations`, not to the reference: see the docstring. This is
        # the difference between a longer run that improves geometry and one that
        # spends its extra hour with the positions effectively frozen.
        "position_lr_max_steps": iterations,
        # Lowered so a room-scale scene gets primitives in proportion to its surface.
        "densify_grad_threshold": policy.paper_densify_grad_threshold
        * policy.densify_grad_multiple,
        "densify_abs_grad_threshold": policy.paper_densify_abs_grad_threshold
        * policy.densify_grad_multiple,
        # --- densification cadence: deliberately NOT scaled ---
        #
        # `densification_interval` stays at the paper's 100 and is passed
        # explicitly so it is visible in the log rather than inherited silently.
        #
        # Scaling it with the schedule is the obvious move and it is wrong. The
        # multi-view trim it has to out-pace is hardcoded `iteration % 1000` at
        # `train.py:418` — not an argument, so it cannot be scaled with the run.
        # The paper's 30k schedule densifies every 100 and trims every 1,000: ten
        # densify steps per trim. Stretching the interval to 500 on a 98k run
        # would leave 98 densify steps against 49 trims, a 2:1 ratio, quietly
        # *reducing* net growth. Against a fixed absolute trim, the interval must
        # stay absolute too.
        #
        # This is also why the 98k run's stall (5,986,377 -> 5,992,634 between 23k
        # and 98k, +0.1%) is NOT explained by the cadence: at interval 100 and
        # `densify_until_iter` 49,000 it already had 490 densify steps against 49
        # trims, the paper's ratio. The likelier cause is the gradient gate --
        # as the fit converges fewer gaussians exceed `densify_grad_threshold` --
        # but with `total_points` logged only twice that is inference, not
        # measurement, which is what the denser `test_iterations` below fix.
        "densification_interval": 100,
        # So the run reports a number at its own end rather than at 30,000, and
        # often enough in between that the density curve is observable instead of
        # inferred: the 98k run logged `total_points` exactly twice, which is why
        # the stall above had to be diagnosed from source rather than measured.
        "test_iterations": sorted({at(7_000), *range(10_000, iterations, 10_000), iterations}),
        "save_iterations": sorted({iterations}),
        "evidence": {
            "images": n_images,
            "camera_step_median": step_median,
            "camera_extent": extent,
            "views_per_image": iterations / max(n_images, 1),
            "reference_views_per_image": policy.reference_iterations / policy.reference_images,
            "schedule_scale": scale,
            # The paper's own numbers, so the derivation can be sanity-checked
            # against the setting they were tuned in.
            "paper_min_dis": 0.01,
            "paper_max_dis": 1.5,
            "paper_iterations": policy.reference_iterations,
            "paper_position_lr_max_steps": 30_000,
            "paper_densify_grad_threshold": policy.paper_densify_grad_threshold,
            "densify_grad_multiple": policy.densify_grad_multiple,
            "pilot_gaussians_smaller_room": 1_247_783,
            "this_run_gaussians_before_change": 636_362,
            "derived_over_paper_min_dis": min_dis / 0.01,
            "derived_over_paper_max_dis": max_dis / 1.5,
        },
    }


def prune_components(mesh, policy: KnobPolicy | None = None) -> dict:
    """Drop mesh components below a fraction of the largest, in place.

    This is the half of the cleanup that `num_cluster=1` got wrong. Size relative
    to the biggest component is the right criterion: it removes speckle at any
    scene scale and keeps a crate that the fusion never welded to the floor.
    Returns what it removed, because a cleanup that does not report its own
    losses is indistinguishable from a cleanup that removed the subject.
    """
    policy = policy or KnobPolicy()
    import open3d as o3d

    idx, counts, _ = mesh.cluster_connected_triangles()
    idx = np.asarray(idx)
    counts = np.asarray(counts)
    if counts.size == 0:
        return {"components": 0, "kept": 0, "removed": 0}

    floor = max(1, int(policy.keep_cluster_frac * counts.max()))
    keep_ids = np.argsort(counts)[::-1][: policy.max_clusters]
    keep_ids = np.array([i for i in keep_ids if counts[i] >= floor])
    doomed = ~np.isin(idx, keep_ids)
    removed_tris = int(doomed.sum())
    mesh.remove_triangles_by_mask(doomed)
    mesh.remove_unreferenced_vertices()

    return {
        "components": int(counts.size),
        "kept": int(len(keep_ids)),
        "removed": int(counts.size - len(keep_ids)),
        "triangles_removed": removed_tris,
        "triangles_kept": int(counts.sum() - removed_tris),
        "largest_component_triangles": int(counts.max()),
        "size_floor_triangles": floor,
        # What num_cluster=1 would have thrown away, in triangles. This is the
        # number that says whether the inherited constant was costing objects.
        "triangles_beyond_largest_component": int(counts.sum() - counts.max()),
    }


def _to4(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    if m.shape == (4, 4):
        return m
    out = np.eye(4)
    out[:3, :] = m
    return out
