import torch

from aml_gnn import artifacts
from aml_gnn.data import TransactionGraph
from aml_gnn.evaluate import evaluate_split
from aml_gnn.model import MultiPNA, model_args_for
from aml_gnn.sampler import GraphView
from aml_gnn.train import fit


def tiny_graph(num_nodes=50, num_edges=400, num_feats=6) -> TransactionGraph:
    g = torch.Generator().manual_seed(0)
    return TransactionGraph(
        num_nodes=num_nodes,
        src=torch.randint(0, num_nodes, (num_edges,), generator=g),
        dst=torch.randint(0, num_nodes, (num_edges,), generator=g),
        attr=torch.randn(num_edges, num_feats, generator=g),
        time=torch.arange(num_edges, dtype=torch.float32),
        y=(torch.rand(num_edges, generator=g) < 0.1).long(),
        split=torch.cat([torch.zeros(240), torch.ones(80), torch.full((80,), 2)]).long(),
    )


def test_sampler_returns_targets_and_only_visible_edges():
    graph = tiny_graph()
    view = GraphView(graph, graph.split <= 0, num_neighbors=[5, 5])
    target = graph.edges_in(0)[:16]
    x, edge_index, edge_attr, pos = view.batch(target)

    assert edge_attr.shape == (edge_index.shape[1], graph.attr.shape[1] + 1)
    assert torch.equal(edge_attr[pos, :-1], graph.attr[target])       # pos points at the targets
    assert x[edge_index[:, pos].flatten(), 1].eq(1).all()             # ego flag on their accounts
    train_attr = graph.attr[graph.split == 0]
    for row in edge_attr[:, :-1]:                                     # no val/test edge leaks in
        assert (train_attr == row).all(1).any()


def test_model_trains_and_evaluates():
    torch.manual_seed(0)
    graph = tiny_graph()
    model = MultiPNA(**model_args_for(graph, graph.edges_in(0)))
    logits = model(*GraphView(graph, graph.split <= 0, [5, 5]).batch(graph.edges_in(0)[:32]))
    assert logits.shape == (32, 2)
    logits.sum().backward()

    metrics = evaluate_split(model, GraphView(graph, graph.split <= 1, [5, 5]),
                             graph.edges_in(1), graph.y, batch_size=32)
    assert set(metrics) == {"precision", "recall", "f1", "auprc"}


def test_fit_respects_batch_cap_and_time_budget():
    torch.manual_seed(0)
    graph = tiny_graph()
    model = MultiPNA(**model_args_for(graph, graph.edges_in(0)))
    views = [GraphView(graph, graph.split <= s, [5, 5]) for s in range(3)]
    targets = [graph.edges_in(s) for s in range(3)]

    result = fit(model, views, targets, graph.y, epochs=50, patience=50, batch_size=32,
                 train_batches=2, time_budget=1e-6)
    assert result.stopped_by == "time_budget" and len(result.history) == 1
    assert result.history[0]["train_edges"] == 64          # 2 batches × 32, not all 240
    assert result.best_state is not None


def test_checkpoint_roundtrip(tmp_path):
    graph = tiny_graph()
    args = model_args_for(graph, graph.edges_in(0))
    model = MultiPNA(**args)
    artifacts.save_checkpoint(model.state_dict(), args, tmp_path / "model.pt")
    loaded = artifacts.load_model(tmp_path / "model.pt")
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
