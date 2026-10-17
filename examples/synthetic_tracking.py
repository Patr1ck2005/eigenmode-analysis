"""Reproducible candidate-path and grid diagnostics using the stable public API.

Run from the repository: python examples/synthetic_tracking.py --out docs/assets/showcase
Only synthetic scalar functions are used. Matplotlib is a documentation dependency.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from eigenmode_analysis.tracking import CandidateSet, TrackingConfig, track_path, track_graph, rectilinear_traversal
from _showcase import COLORS, STYLE, Showcase, plt


def path_case():
    rng = np.random.default_rng(1729)
    x = np.linspace(-1, 1, 91)
    values = np.column_stack((1 + .28*x, 1 - .28*x, 1.7 + .08*np.cos(np.pi*x)))
    points = []
    for i, row in enumerate(values):
        order = rng.permutation(3)
        order = order[order != 2] if 56 <= i < 64 else order
        points.append(CandidateSet([f"sample-{i}-{j}" for j in order], row[order, None]))
    result = track_path(points, coordinates=x, config=TrackingConfig(n_tracks=3, derivative_weights=2))
    for i, ids in enumerate(result.candidate_ids):
        assigned = [v for v in ids if v is not None]
        assert set(assigned) == set(points[i].ids), "Raw candidate identity was lost or duplicated"
        np.testing.assert_allclose(np.sort(result.features[i, np.isfinite(result.features[i, :, 0]), 0]), np.sort(points[i].features[:, 0]))
    assert np.isnan(result.features[56:64, 2, 0]).all()
    assert sum(v is None for row in result.candidate_ids for v in row) == 8
    np.testing.assert_allclose(result.features[:56, :2, 0], values[:56, :2], atol=1e-12)
    return x, points, result


def graph_case():
    x, y = np.linspace(-1, 1, 13), np.linspace(-.7, .7, 9)
    X, Y = np.meshgrid(x, y, indexing="ij")
    fields = np.stack((.8+.12*X+.09*Y**2, 1.5+.08*X**2-.07*Y), axis=-1)
    points = {}
    for i, j in np.ndindex(X.shape):
        order = [1, 0] if (i+j)%2 else [0, 1]
        points[i, j] = CandidateSet(order, fields[i,j,order,None]) if (i,j) != (5,4) else CandidateSet([], np.empty((0,1)))
    order, edges = rectilinear_traversal(X.shape, (x[1]-x[0], y[1]-y[0]))
    result = track_graph(points, node_order=order, edges=edges, anchor_id=(0,0), config=TrackingConfig(n_tracks=2))
    missing = result.node_ids.index((5,4))
    assert result.candidate_ids[missing] == (None,None)
    assert result.coverage == (X.size-1)/X.size
    assert not result.unreachable_nodes
    reverse = {node: CandidateSet(tuple(reversed(p.ids)), p.features[::-1]) for node,p in points.items()}
    check = track_graph(reverse, node_order=order, edges=edges, anchor_id=(0,0), config=TrackingConfig(n_tracks=2))
    assert result.candidate_ids == check.candidate_ids
    return x, y, result


def draw_path(ax, x, points, result, tracked):
    if tracked:
        for j in range(3):
            ax.plot(x, result.features[:,j,0], color=COLORS[j], lw=1, marker=".", ms=2.4, label=f"Track {j+1}")
    else:
        ax.scatter(np.repeat(x, [p.n_candidates for p in points]), np.concatenate([p.features[:,0] for p in points]), s=7, c="#53657b", alpha=.8)
    ax.set(xlim=(-1.03,1.03), ylim=(.65,1.85), xlabel="Path coordinate t", ylabel="Synthetic eigenvalue")
    ax.set_title("Assigned identities" if tracked else "Unordered candidates", loc="left")
    if tracked:
        ax.axvspan(x[56],x[63],color="#d7e1ec",alpha=.5,zorder=0)
        ax.legend(frameon=False, fontsize=7, loc="center", bbox_to_anchor=(.25,.58))


def main(out):
    x, points, result = path_case()
    gx, gy, graph = graph_case()
    show = Showcase(out, "Eigenmode Analysis · synthetic tracking", __file__,
        "All coordinates and eigenvalues are dimensionless. Both path panels show exactly the same candidates. Colors denote computed track labels; shaded interval contains eight missing samples of the upper branch. Lines connect available samples only; no gap is filled. The graph uses a 13 × 9 non-square grid and one empty site.",
        [{"intrinsic_dimension":"1D", "sampling_topology":"ordered path with shuffled candidate lists", "parameter_space":"generic scalar parameter", "quantity":"synthetic eigenvalue", "compatible_views":["scatter","tracked lines","assignment diagnostics"], "selected_views":"raw/tracked comparison and cost diagnostic", "transformation_status":"same-dimensional assignment, no interpolation"},
         {"intrinsic_dimension":"2D", "sampling_topology":"regular non-square grid with one hole", "parameter_space":"generic two-parameter scan", "quantity":"tracked scalar values and assignment coverage", "compatible_views":["heatmap","surface"], "selected_views":"coverage and cost heatmaps", "transformation_status":"graph assignment"}])
    with plt.rc_context(STYLE):
        fig,axs=plt.subplots(1,2,figsize=(210/25.4,75/25.4),layout="constrained")
        draw_path(axs[0],x,points,result,False);draw_path(axs[1],x,points,result,True)
        show.save(fig,"hero","Same samples, explicit identities","Hungarian path tracking resolves the crossing using continuity; the missing upper-branch interval remains absent.")
        fig,ax=plt.subplots(figsize=(105/25.4,70/25.4),layout="constrained")
        draw_path(ax,x,points,result,True)
        show.save(fig,"crossing-missing","Crossing and missing candidates","Candidate IDs are local opaque strings. All 265 observations survive assignment, with eight missing track slots.")
        fig,axs=plt.subplots(2,1,figsize=(105/25.4,95/25.4),sharex=True,layout="constrained")
        for j in range(3): axs[0].plot(x,result.costs[:,j],color=COLORS[j],label=f"Track {j+1}")
        axs[0].set(ylabel="Assignment cost");axs[0].legend(frameon=False,ncol=3,fontsize=7)
        assigned=np.sum(np.isfinite(result.features[:,:,0]),axis=1)
        axs[1].step(x,assigned,where="mid",color=COLORS[0]);axs[1].set(xlabel="Path coordinate t",ylabel="Assigned / site",ylim=(0,3.3),yticks=(0,1,2,3))
        show.save(fig,"path-diagnostics","Assignment diagnostics","Costs are numerical matching costs, not probabilities. Missing samples reduce coverage without inventing replacement values.")
        fig,axs=plt.subplots(1,2,figsize=(210/25.4,75/25.4),layout="constrained")
        coverage=np.array([sum(c is not None for c in row) for row in graph.candidate_ids]).reshape(len(gx),len(gy))
        has_cost=np.isfinite(graph.costs).any(axis=1)
        cost=np.full(graph.n_nodes,np.nan)
        cost[has_cost]=np.nanmax(graph.costs[has_cost],axis=1)
        cost=cost.reshape(len(gx),len(gy))
        for ax,z,title,cmap in zip(axs,(coverage,cost),("Assigned candidates / site","Maximum assignment cost"),("Blues","magma")):
            mesh=ax.pcolormesh(gx,gy,z.T,shading="nearest",cmap=cmap)
            fig.colorbar(mesh,ax=ax,shrink=.8);ax.set(xlabel="x",ylabel="y",title=title,aspect="equal")
        show.save(fig,"graph-diagnostics","Directed-graph tracking","The empty site is retained; all other sites have two assignments. White in the cost map also includes the anchor, which has no incoming assignment cost. Reversing candidate order leaves identities unchanged.")
    show.finish({"raw_observations":sum(p.n_candidates for p in points),"assigned_observations":int(np.isfinite(result.features[:,:,0]).sum()),"missing_path_slots":8,"crossing_expected_values_match":True,"graph_coverage":graph.coverage,"graph_order_invariance":True,"graph_conflicts":len(graph.conflicts),"all_assertions_passed":True})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,default=Path("docs/assets/showcase"));main(parser.parse_args().out)
