import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler as prof

import dgl
import dgl.data
from dgl.nn import GraphConv

os.environ["DGLBACKEND"] = "pytorch"


def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    for e in range(100):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print(
                f"In epoch {e}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f})"
            )


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


def latency_profiler_wrapper(fun):
    """Pytorch Profiler wrapper for what we want to measure"""

    def wrapped(*args, **kwargs):
        # Setup pytorch profiler
        with prof.profile(
            activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as p:
            # Payload to profile
            ret = fun(*args, **kwargs)

        # Save trace to visualize with Perfetto UI
        p.export_chrome_trace(path="latency_chrome_trace.json")

        # Dump key averages to console (sorted by CUDA total time)
        print(
            p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
        )
        return ret

    return wrapped


def execute(device="cpu", seed=1337):
    torch.manual_seed(seed)
    print(f"Training GCN on {device}")
    dataset = dgl.data.CoraGraphDataset()
    print(f"Number of categories: {dataset.num_classes}")
    ds = dataset[0].to(device)
    model = GCN(ds.ndata["feat"].shape[1], 16, dataset.num_classes).to(device)
    train(ds, model)
    print()
    return model


def compare_tensors(t1, t2, *args, **kwargs):
    context = 4
    full_print_limit = 64
    torch.set_printoptions(linewidth=360)

    t1 = t1.cpu()
    t2 = t2.cpu()
    if t1.dtype != t2.dtype:
        print("Dtypes don't match")
        return False
    if t1.shape != t2.shape:
        print("Shapes don't match")
        return False
    if torch.allclose(t1, t2, *args, **kwargs):
        return True

    print("Values don't match")
    close = torch.isclose(t1, t2, *args, **kwargs).to(torch.int32)
    first_diff_idx = torch.argmin(close)
    el_count = t1.numel()
    if el_count <= full_print_limit:
        slice_idxs = slice(el_count)
    else:
        first_print_idx = max(first_diff_idx - context, 0)
        slice_idxs = slice(first_print_idx, first_print_idx + 2 * context)
    stack = torch.stack(
        (
            torch.arange(el_count, dtype=t1.dtype),
            t1.flatten(),
            t2.flatten(),
            (t1 - t2).flatten(),
        )
    )
    print(f"First difference at {first_diff_idx}. idx, t1, t2, t1-t2")
    print(stack[:, slice_idxs])
    return False


if __name__ == "__main__":
    model_cpu = execute()
    model_gpu = latency_profiler_wrapper(execute)("cuda")

    for p_cpu, p_gpu in zip(model_cpu.parameters(), model_gpu.parameters()):
        assert p_cpu.data.device.type == "cpu", f"{p_cpu.data.device}"
        assert p_gpu.data.device.type == "cuda", f"{p_gpu.data.device}"
        if not compare_tensors(p_cpu.data, p_gpu.data, rtol=1e-3, atol=1e-3):
            print("Parameter mismatch")
            sys.exit(1)
    else:
        print("All params match")
