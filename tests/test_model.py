from dataclasses import replace

import pytest
import torch

from revllm.model import ModelConfig, TinyGPT, count_unique_parameters
from revllm.correctness import reconstruction_probe


@pytest.mark.parametrize("kind", ["conventional", "midpoint", "coupled_euler", "blended_midpoint"])
def test_full_model_parameter_count(kind):
    model = TinyGPT(ModelConfig(integrator=kind))
    assert count_unique_parameters(model) == 19_969_152
    assert model.lm_head.weight is model.token_embedding.weight


@pytest.mark.parametrize("kind", ["midpoint", "coupled_euler", "blended_midpoint"])
@pytest.mark.parametrize("depth", [2, 4])
def test_reconstructed_matches_stored_loss_input_and_every_parameter_gradient(kind, depth):
    cfg = ModelConfig(
        vocab_size=32, block_size=8, n_layer=depth, n_head=2, n_embd=16,
        integrator=kind, step_size=0.25, blend=0.5,
    )
    torch.manual_seed(9182)
    stored = TinyGPT(cfg).double()
    reconstructed = TinyGPT(replace(cfg, backward_mode="reconstructed")).double()
    reconstructed.load_state_dict(stored.state_dict())
    ids = torch.randint(0, 32, (2, 5))
    target = torch.randint(0, 32, (2, 5))
    loss_a = stored(ids, target)
    loss_b = reconstructed(ids, target)
    torch.testing.assert_close(loss_a, loss_b, atol=1e-11, rtol=1e-11)
    loss_a.backward()
    loss_b.backward()
    grads_a = dict(stored.named_parameters())
    grads_b = dict(reconstructed.named_parameters())
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        torch.testing.assert_close(grads_a[name].grad, grads_b[name].grad, atol=1e-9, rtol=1e-8, msg=name)


def test_future_token_cannot_affect_earlier_logits():
    model = TinyGPT(ModelConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2, n_embd=16)).eval()
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 3, 5]])
    with torch.no_grad():
        torch.testing.assert_close(model(a)[:, :3], model(b)[:, :3])


def test_ordinary_euler_cannot_be_reversed_by_subtracting_at_output():
    x, h = 2.0, 0.5
    force = lambda z: 3 * z
    output = x + h * force(x)
    incorrect_reconstruction = output - h * force(output)
    assert incorrect_reconstruction != x


def test_reconstructed_input_gradient_matches_stored():
    cfg = ModelConfig(vocab_size=32, block_size=8, n_layer=3, n_head=2, n_embd=16, integrator="coupled_euler")
    torch.manual_seed(11)
    a = TinyGPT(cfg).double()
    b = TinyGPT(replace(cfg, backward_mode="reconstructed")).double()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 32, (2, 4))
    ya = a.token_embedding(x).detach().requires_grad_(True)
    yb = ya.detach().clone().requires_grad_(True)
    loss_a = a.stack(ya).square().sum()
    loss_b = b.stack(yb).square().sum()
    loss_a.backward()
    loss_b.backward()
    torch.testing.assert_close(ya.grad, yb.grad, atol=1e-9, rtol=1e-8)


def test_reconstructed_gradient_matches_finite_difference():
    cfg = ModelConfig(vocab_size=16, block_size=4, n_layer=2, n_head=2,
                      n_embd=8, integrator="midpoint", backward_mode="reconstructed")
    torch.manual_seed(7)
    model = TinyGPT(cfg).double()
    x = torch.tensor([[2, 3, 4, 5]])
    y = torch.tensor([[3, 4, 5, 6]])
    (model(x, y) / y.numel()).backward()
    parameter = model.token_embedding.weight
    analytic = parameter.grad[2, 0].item()
    original = parameter[2, 0].item()
    epsilon = 1e-5
    with torch.no_grad():
        parameter[2, 0] = original + epsilon
    positive = (model(x, y) / y.numel()).item()
    with torch.no_grad():
        parameter[2, 0] = original - epsilon
    negative = (model(x, y) / y.numel()).item()
    with torch.no_grad():
        parameter[2, 0] = original
    numerical = (positive - negative) / (2 * epsilon)
    assert abs(analytic - numerical) <= 1e-6 * max(1, abs(numerical))


def test_reconstructed_forward_saved_activations_do_not_scale_with_depth():
    def saved_nonparameter_elements(depth: int, mode: str) -> int:
        cfg = ModelConfig(vocab_size=32, block_size=8, n_layer=depth, n_head=2,
                          n_embd=16, integrator="midpoint", backward_mode=mode)
        model = TinyGPT(cfg)
        parameter_ptrs = {p.data_ptr() for p in model.parameters()}
        seen = []

        def pack(tensor):
            if tensor.data_ptr() not in parameter_ptrs:
                seen.append(tensor.numel())
            return tensor

        x = torch.randn(2, 8, 16, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            model.stack(x).square().sum()
        return sum(seen)

    reconstructed_2 = saved_nonparameter_elements(2, "reconstructed")
    reconstructed_4 = saved_nonparameter_elements(4, "reconstructed")
    stored_2 = saved_nonparameter_elements(2, "stored")
    stored_4 = saved_nonparameter_elements(4, "stored")
    assert reconstructed_2 == reconstructed_4
    assert stored_4 > stored_2 > reconstructed_4


@pytest.mark.parametrize("kind", ["midpoint", "coupled_euler", "blended_midpoint"])
@pytest.mark.parametrize("dtype,threshold", [(torch.float64, 1e-9), (torch.float32, 1e-5)])
def test_state_reconstruction_error_gate(kind, dtype, threshold):
    cfg = ModelConfig(vocab_size=32, block_size=8, n_layer=6, n_head=2, n_embd=16,
                      integrator=kind, backward_mode="reconstructed", blend=0.5)
    model = TinyGPT(cfg).to(dtype=dtype)
    result = reconstruction_probe(model)
    assert result["threshold"] == threshold
    assert result["status"] == "PASS"


def test_one_batch_overfits():
    torch.manual_seed(101)
    model = TinyGPT(ModelConfig(vocab_size=16, block_size=4, n_layer=2,
                                n_head=2, n_embd=16, integrator="coupled_euler",
                                backward_mode="reconstructed"))
    x = torch.tensor([[2, 3, 4, 5]])
    y = torch.tensor([[3, 4, 5, 6]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    first = None
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y) / y.numel()
        if first is None:
            first = float(loss.detach())
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < first * 0.5
