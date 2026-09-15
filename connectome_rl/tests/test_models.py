"""Unit tests for Model Architectures.

Tests:
1. Standard Dense MLP baseline (control model).
2. MaskedLinear layer (biological synaptic constraint & gradient isolation).
3. ConnectomePolicy (feedforward biological connectome Actor-Critic network).
4. ConnectomeRNNPolicy (recurrent biological connectome with CPG rhythm loops).
5. SynapticGNNPolicy (message passing on the 1,581-synapse biological graph).
"""

from pathlib import Path
import pytest
import torch
import numpy as np

from connectome_rl.src.models.mlp_policy import MLPPolicy
from connectome_rl.src.models.connectome_layers import MaskedLinear
from connectome_rl.src.models.connectome_policy import ConnectomePolicy
from connectome_rl.src.models.connectome_rnn import ConnectomeRNNPolicy
from connectome_rl.src.models.synaptic_gnn import SynapticGNNPolicy
from connectome_rl.src.models.hierarchical_policy import HierarchicalPolicy


class TestMLPPolicy:
    """Test suite for the unconstrained MLPPolicy baseline model."""

    @pytest.fixture
    def policy_setup(self):
        obs_dim = 100
        act_dim = 42
        batch_size = 8
        policy = MLPPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=128)
        dummy_obs = torch.randn(batch_size, obs_dim)
        return policy, dummy_obs, obs_dim, act_dim, batch_size

    def test_initialization_and_shapes(self, policy_setup):
        """Plain English: Does the brain start with 42 calm leg outputs near 0.0?"""
        policy, dummy_obs, obs_dim, act_dim, batch_size = policy_setup
        assert policy.obs_dim == obs_dim
        assert policy.act_dim == act_dim
        assert policy.actor_logstd.shape == (act_dim,)
        
        with torch.no_grad():
            initial_mean = policy.get_deterministic_action(dummy_obs)
            assert initial_mean.abs().mean() < 0.2, "Initial actions should be near zero for stability"

    def test_action_sampling(self, policy_setup):
        """Plain English: Can the brain generate random exploratory steps without producing NaNs?"""
        policy, dummy_obs, _, act_dim, batch_size = policy_setup
        action, log_prob, entropy, value = policy.get_action_and_value(dummy_obs)
        
        assert action.shape == (batch_size, act_dim)
        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size, 1)
        
        assert not torch.isnan(action).any()
        assert not torch.isnan(log_prob).any()
        assert not torch.isnan(entropy).any()
        assert not torch.isnan(value).any()

    def test_evaluating_specific_action(self, policy_setup):
        """Plain English: When evaluating a past step, does the math accurately compute its probability?"""
        policy, dummy_obs, _, act_dim, batch_size = policy_setup
        given_action = torch.randn(batch_size, act_dim)
        
        action, log_prob, entropy, value = policy.get_action_and_value(dummy_obs, action=given_action)
        
        assert torch.allclose(action, given_action)
        assert log_prob.shape == (batch_size,)
        assert not torch.isnan(log_prob).any()

    def test_deterministic_actions(self, policy_setup):
        """Plain English: Without random noise, does the fly make the identical step for the identical stance?"""
        policy, dummy_obs, _, act_dim, batch_size = policy_setup
        act1 = policy.get_deterministic_action(dummy_obs)
        act2 = policy.get_deterministic_action(dummy_obs)
        
        assert act1.shape == (batch_size, act_dim)
        assert torch.allclose(act1, act2)

    def test_value_function(self, policy_setup):
        """Plain English: Does the Critic output one single quality score per fly?"""
        policy, dummy_obs, _, _, batch_size = policy_setup
        v = policy.get_value(dummy_obs)
        assert v.shape == (batch_size, 1)
        assert not torch.isnan(v).any()

    def test_gradient_flow_and_backpropagation(self, policy_setup):
        """Plain English: Does learning feedback reach every single weight in the network?"""
        policy, dummy_obs, _, act_dim, batch_size = policy_setup
        action, log_prob, entropy, value = policy.get_action_and_value(dummy_obs)
        
        dummy_advantages = torch.ones(batch_size)
        dummy_returns = torch.ones(batch_size, 1)
        
        actor_loss = -(log_prob * dummy_advantages).mean() - 0.01 * entropy.mean()
        critic_loss = ((value - dummy_returns) ** 2).mean()
        total_loss = actor_loss + 0.5 * critic_loss
        
        total_loss.backward()
        
        assert policy.actor_logstd.grad is not None
        assert not torch.isnan(policy.actor_logstd.grad).any()
        
        for name, param in policy.named_parameters():
            assert param.grad is not None, f"Parameter {name} did not receive gradients"
            assert not torch.isnan(param.grad).any(), f"Gradient in {name} is NaN"


class TestMaskedLinear:
    """Test suite for the core MaskedLinear connectome layer."""

    @pytest.fixture
    def toy_masked_layer(self):
        mask = torch.tensor([
            [True,  False, False, False],
            [True,  True,  False, False],
            [False, True,  False, False],
            [False, False, True,  False],
            [False, False, True,  True ],
            [False, False, False, True ],
        ], dtype=torch.bool)
        layer = MaskedLinear(in_features=4, out_features=6, mask=mask)
        return layer, mask

    def test_forward_weights_are_strictly_zero_where_unwired(self, toy_masked_layer):
        """Plain English: Are non-existent biological connections guaranteed to be 0.0 in forward pass?"""
        layer, mask = toy_masked_layer
        eff_w = layer.effective_weight
        assert (eff_w[~mask] == 0.0).all(), "Unwired weights must be strictly 0.0!"

    def test_backward_gradient_isolation(self, toy_masked_layer):
        """Plain English: Does the backward hook guarantee that unwired connections NEVER receive gradients?"""
        layer, mask = toy_masked_layer
        x = torch.randn(2, 4)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        grad = layer.weight.grad
        assert (grad[~mask] == 0.0).all(), "Gradients at un-wired synapses must be strictly 0.0!"
        assert (grad[mask] != 0.0).all(), "Active synapses must receive real gradients!"

    def test_in_silico_lesion_and_recovery(self, toy_masked_layer):
        """Plain English: Can we mathematically 'cut' a neuron and then restore it back to health?"""
        layer, mask = toy_masked_layer
        initial_active = layer.effective_mask.sum().item()
        
        silenced = layer.apply_lesion(target_rows=[1])
        assert silenced > 0
        assert layer.effective_mask.sum().item() == initial_active - silenced
        
        layer.reset_lesions()
        assert layer.effective_mask.sum().item() == initial_active


class TestConnectomePolicy:
    """Test suite for the full feedforward ConnectomePolicy."""

    @pytest.fixture
    def real_circuit_policy(self):
        data_path = Path(__file__).resolve().parents[1] / "data" / "dna_circuit_tensors.pt"
        if not data_path.exists():
            pytest.skip("dna_circuit_tensors.pt not found on disk")
            
        obs_dim = 100
        act_dim = 42
        policy = ConnectomePolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            circuit_data=data_path,
            use_biological_weights=True,
        )
        dummy_obs = torch.randn(4, obs_dim)
        return policy, dummy_obs, obs_dim, act_dim

    def test_connectome_policy_shapes_and_hierarchy(self, real_circuit_policy):
        """Plain English: Does the signal flow cleanly: 100 sensory inputs -> 4 DNs -> 125 INs -> 377 MNs -> 42 leg joints?"""
        policy, dummy_obs, _, act_dim = real_circuit_policy
        
        action_mean, activations = policy.forward_actor(dummy_obs)
        assert action_mean.shape == (4, act_dim)
        
        assert activations["dn"].shape == (4, 4), "Should have 4 Descending Neurons"
        assert activations["interneuron"].shape == (4, 125), "Should have 125 VNC Interneurons"
        assert activations["motor_neuron"].shape == (4, 377), "Should have 377 Motor Neurons"

    def test_connectome_policy_biological_gradient_isolation(self, real_circuit_policy):
        """Plain English: When training the whole fly, are unwired connections across the whole brain protected from updates?"""
        policy, dummy_obs, _, _ = real_circuit_policy
        action, log_prob, entropy, value = policy.get_action_and_value(dummy_obs)
        
        loss = value.mean() - log_prob.mean()
        loss.backward()
        
        hop1_unwired_grad = policy.hop1_layer.weight.grad[~policy.hop1_layer.mask]
        assert (hop1_unwired_grad == 0.0).all(), "Hop 1 unwired synapses received gradients!"
        
        hop2_unwired_grad = policy.hop2_layer.weight.grad[~policy.hop2_layer.mask]
        assert (hop2_unwired_grad == 0.0).all(), "Hop 2 unwired synapses received gradients!"

    def test_connectome_policy_in_silico_lesion(self, real_circuit_policy):
        """Plain English: When we 'cut' the DNa01 steering neuron, does it noticeably alter the fly's leg commands?"""
        policy, dummy_obs, _, _ = real_circuit_policy
        
        intact_actions = policy.get_deterministic_action(dummy_obs)
        silenced = policy.lesion_dn_channel(0)
        assert silenced > 0, "Silencing DNa01 should cut downstream synapses"
        
        lesioned_actions = policy.get_deterministic_action(dummy_obs)
        diff = (lesioned_actions - intact_actions).abs().mean().item()
        assert diff > 0.0, "Lesioning DNa01 must alter leg motor commands"
        
        policy.reset_lesions()
        restored_actions = policy.get_deterministic_action(dummy_obs)
        assert torch.allclose(restored_actions, intact_actions), "Resetting lesions must restore 100% normal function"


class TestConnectomeRNNPolicy:
    """Test suite for the recurrent ConnectomeRNNPolicy (CPG model)."""

    @pytest.fixture
    def rnn_policy_setup(self):
        data_path = Path(__file__).resolve().parents[1] / "data" / "dna_circuit_tensors.pt"
        if not data_path.exists():
            pytest.skip("dna_circuit_tensors.pt not found on disk")
            
        obs_dim = 100
        act_dim = 42
        policy = ConnectomeRNNPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            circuit_data=data_path,
        )
        dummy_obs = torch.randn(2, obs_dim)
        return policy, dummy_obs, obs_dim, act_dim

    def test_connectome_rnn_shapes_and_state_propagation(self, rnn_policy_setup):
        """Plain English: Can the recurrent VNC cord propagate a 125-dim hidden memory state across sequential stepping time?"""
        policy, dummy_obs, _, act_dim = rnn_policy_setup
        h = policy.init_hidden(batch_size=2)
        assert h.shape == (2, 125), "Initial hidden state must be (batch_size, 125)"
        
        for step in range(4):
            action, log_prob, entropy, value, h_next = policy.get_action_and_value(dummy_obs, h_prev=h)
            assert action.shape == (2, act_dim)
            assert h_next.shape == (2, 125)
            assert not torch.isnan(h_next).any()
            assert not torch.isnan(action).any()
            assert not torch.allclose(h, h_next)
            h = h_next

    def test_connectome_rnn_recurrent_gradient_isolation(self, rnn_policy_setup):
        """Plain English: Do non-existent lateral loops between interneurons stay at 0.0 gradient during recurrent learning?"""
        policy, dummy_obs, _, _ = rnn_policy_setup
        h = policy.init_hidden(batch_size=2)
        action, log_prob, entropy, value, h = policy.get_action_and_value(dummy_obs, h_prev=h)
        
        loss = value.mean() - log_prob.mean()
        loss.backward()
        
        rec_layer = policy.rnn_cell.rec_layer
        unwired_rec_grad = rec_layer.weight.grad[~rec_layer.mask]
        assert (unwired_rec_grad == 0.0).all(), "Unwired recurrent synapses received non-zero gradients!"

    def test_connectome_rnn_cpg_lesion_and_recovery(self, rnn_policy_setup):
        """Plain English: When we surgically silence the 82 biological CPG rhythm loops, does it silence all 82 synapses and cleanly recover?"""
        policy, dummy_obs, _, _ = rnn_policy_setup
        initial_rec_active = policy.rnn_cell.rec_layer.effective_mask.sum().item()
        assert initial_rec_active == 82, "Should have exactly 82 active recurrent CPG synapses"
        
        silenced = policy.lesion_cpg_loops()
        assert silenced == 82
        assert policy.rnn_cell.rec_layer.effective_mask.sum().item() == 0
        
        policy.reset_lesions()
        assert policy.rnn_cell.rec_layer.effective_mask.sum().item() == 82


class TestSynapticGNNPolicy:
    """Test suite for the Graph Neural Network SynapticGNNPolicy."""

    @pytest.fixture
    def gnn_policy_setup(self):
        data_path = Path(__file__).resolve().parents[1] / "data" / "dna_circuit_tensors.pt"
        if not data_path.exists():
            pytest.skip("dna_circuit_tensors.pt not found on disk")
            
        obs_dim = 100
        act_dim = 42
        policy = SynapticGNNPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            circuit_data=data_path,
            node_dim=16,
            num_hops=2,
        )
        dummy_obs = torch.randn(2, obs_dim)
        return policy, dummy_obs, obs_dim, act_dim

    def test_synaptic_gnn_shapes_and_message_passing(self, gnn_policy_setup):
        """Plain English: Does information successfully propagate along all 1,581 biological synapses across 2 hops to produce 42 leg commands?"""
        policy, dummy_obs, _, act_dim = gnn_policy_setup
        action, log_prob, entropy, value = policy.get_action_and_value(dummy_obs)
        
        assert action.shape == (2, act_dim)
        assert log_prob.shape == (2,)
        assert entropy.shape == (2,)
        assert value.shape == (2, 1)
        
        assert not torch.isnan(action).any()
        assert not torch.isnan(value).any()

    def test_synaptic_gnn_gradient_flow(self, gnn_policy_setup):
        """Plain English: When we run backpropagation through the GNN, do learning gradients cleanly reach all 506 neurons and message layers?"""
        policy, dummy_obs, _, _ = gnn_policy_setup
        action, log_prob, entropy, value = policy.get_action_and_value(dummy_obs)
        
        loss = value.mean() - log_prob.mean()
        loss.backward()
        
        # Base node embeddings and message weights must have valid gradients
        assert policy.base_node_embeddings.grad is not None
        assert not torch.isnan(policy.base_node_embeddings.grad).any()
        
        for gnn_layer in policy.gnn_layers:
            assert gnn_layer.msg_linear.weight.grad is not None
            assert not torch.isnan(gnn_layer.msg_linear.weight.grad).any()

    def test_synaptic_gnn_in_silico_lesion_and_recovery(self, gnn_policy_setup):
        """Plain English: When we sever all incoming and outgoing synapses of the DNa01 steering neuron, does it alter leg commands, and does reset restore normal function?"""
        policy, dummy_obs, _, _ = gnn_policy_setup
        
        intact_action = policy.get_deterministic_action(dummy_obs)
        
        # Sever all synapses connected to DNa01
        silenced_edges = policy.lesion_nodes([policy.dn_indices[0]])
        assert silenced_edges > 0, "Silencing DNa01 in GNN must cut connected synapses"
        
        lesioned_action = policy.get_deterministic_action(dummy_obs)
        diff = (lesioned_action - intact_action).abs().mean().item()
        assert diff > 0.0, "Severing DNa01 synapses must alter downstream motor output"
        
        # Restore synapses
        policy.reset_lesions()
        restored_action = policy.get_deterministic_action(dummy_obs)
        assert torch.allclose(restored_action, intact_action), "Resetting lesions must restore 100% normal function"


class TestHierarchicalPolicy:
    """Test suite for the two-tier HierarchicalPolicy (Cephalic Brain -> Connectome VNC)."""

    @pytest.fixture
    def hierarchical_policy_setup(self):
        data_path = Path(__file__).resolve().parents[1] / "data" / "dna_circuit_tensors.pt"
        if not data_path.exists():
            pytest.skip("dna_circuit_tensors.pt not found on disk")
            
        obs_dim = 100
        act_dim = 42
        policy = HierarchicalPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            circuit_data=data_path,
            brain_hidden_dim=64,
            critic_hidden_dim=128,
        )
        dummy_obs = torch.randn(4, obs_dim)
        return policy, dummy_obs, obs_dim, act_dim

    def test_hierarchical_shapes_and_bottleneck(self, hierarchical_policy_setup):
        """Plain English: Does the high-level brain cleanly compress 100 sensory features down to 4 descending commands, and does the biological VNC expand those 4 commands into 42 leg joint targets?"""
        policy, dummy_obs, obs_dim, act_dim = hierarchical_policy_setup
        
        # 1. High-level brain command bottleneck
        z = policy.get_descending_command(dummy_obs)
        assert z.shape == (4, 4), "Brain output must be (batch_size, 4) descending commands"
        assert (-1.0 <= z).all() and (z <= 1.0).all(), "Descending commands must be bounded within [-1, 1]"
        
        # 2. Low-level VNC actuation
        action_mean = policy.act_from_descending_command(z)
        assert action_mean.shape == (4, act_dim), f"VNC output must be (batch_size, {act_dim})"
        
        # 3. Full Actor-Critic step for PPO
        action, log_prob, entropy, value = policy.get_action_and_value(dummy_obs)
        assert action.shape == (4, act_dim)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4, 1)
        
        assert not torch.isnan(action).any()
        assert not torch.isnan(value).any()

    def test_hierarchical_optogenetic_clamping(self, hierarchical_policy_setup):
        """Plain English: If we bypass the brain and manually inject artificial steering commands (like synthetic laser pulses for left vs right turns), does the biological connectome produce distinctly different leg movements?"""
        policy, _, _, act_dim = hierarchical_policy_setup
        
        # Artificial descending command 1: Turn Left (DNa01_L high, DNa01_R low)
        z_left = torch.tensor([[1.0, -1.0, 0.5, 0.5]])
        # Artificial descending command 2: Turn Right (DNa01_L low, DNa01_R high)
        z_right = torch.tensor([[-1.0, 1.0, 0.5, 0.5]])
        
        action_left = policy.act_from_descending_command(z_left)
        action_right = policy.act_from_descending_command(z_right)
        
        assert action_left.shape == (1, act_dim)
        assert action_right.shape == (1, act_dim)
        
        # Turning left vs turning right must command different joint configurations
        joint_diff = (action_left - action_right).abs().sum().item()
        assert joint_diff > 1e-4, "Left vs right descending stimulation must produce distinct joint angles"

    def test_hierarchical_two_stage_freezing(self, hierarchical_policy_setup):
        """Plain English: Can we freeze the low-level biological spinal cord to train only the brain on navigation, or freeze the brain to train only the spinal cord on stepping?"""
        policy, dummy_obs, _, _ = hierarchical_policy_setup
        
        # Freeze VNC: only brain learns
        policy.freeze_vnc()
        for p in policy.vnc.parameters():
            assert not p.requires_grad
        for p in policy.brain.parameters():
            assert p.requires_grad
            
        action, log_prob, entropy, value = policy.get_action_and_value(dummy_obs)
        loss = value.mean() - log_prob.mean()
        loss.backward()
        
        # Brain parameters receive gradients, VNC parameters do not
        brain_grad_found = any(p.grad is not None and (p.grad != 0).any() for p in policy.brain.parameters())
        assert brain_grad_found, "Brain parameters must receive learning gradients"
        for p in policy.vnc.parameters():
            assert p.grad is None, "Frozen VNC parameters must not receive gradients"
            
        # Unfreeze VNC and freeze Brain: only VNC learns
        policy.zero_grad()
        policy.unfreeze_vnc()
        policy.freeze_brain()
        for p in policy.vnc.parameters():
            assert p.requires_grad
        for p in policy.brain.parameters():
            assert not p.requires_grad

    def test_hierarchical_in_silico_lesion_and_recovery(self, hierarchical_policy_setup):
        """Plain English: When we surgically silence descending steering neuron DNa01 in the hierarchical pipeline, does it alter leg movements, and does reset restore 100% normal function?"""
        policy, dummy_obs, _, _ = hierarchical_policy_setup
        
        intact_actions = policy.get_deterministic_action(dummy_obs)
        
        # Silence DNa01 (channel 0)
        silenced = policy.lesion_dn_channel(0)
        assert silenced > 0, "Silencing DNa01 must cut synaptic connections in VNC"
        
        lesioned_actions = policy.get_deterministic_action(dummy_obs)
        diff = (lesioned_actions - intact_actions).abs().mean().item()
        assert diff > 0.0, "Lesioning DNa01 must alter leg motor commands"
        
        # Restore intact biological connectome
        policy.reset_lesions()
        restored_actions = policy.get_deterministic_action(dummy_obs)
        assert torch.allclose(restored_actions, intact_actions), "Resetting lesions must restore 100% normal function"
