"""Unit tests for Phase 0 environment verification."""

import pytest
import numpy as np

def test_python_and_core_libraries():
    import torch
    import mujoco
    import flygym
    import neuprint
    import networkx as nx
    import gymnasium as gym
    import pandas as pd
    import scipy
    import yaml

    assert torch.__version__ is not None
    assert mujoco.__version__ is not None
    assert nx.__version__ is not None
    assert gym.__version__ is not None
    assert pd.__version__ is not None

def test_mujoco_minimal_step():
    import mujoco
    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <body pos="0 0 1">
          <joint type="free"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.time > 0.0

def test_flygym_simulation_step():
    from flygym import Fly, SingleFlySimulation
    from flygym.arena import FlatTerrain
    
    fly = Fly(spawn_pos=(0.0, 0.0, 0.5))
    sim = SingleFlySimulation(fly=fly, arena=FlatTerrain())
    assert len(fly.actuated_joints) == 42, "Expected 42 actuated joints (6 legs * 7 DOFs)"
    
    obs, info = sim.reset()
    assert "joints" in obs
    assert "contact_forces" in obs
    
    action = {"joints": np.zeros(len(fly.actuated_joints))}
    next_obs, reward, terminated, truncated, next_info = sim.step(action)
    assert "joints" in next_obs

def test_neuprint_connection():
    import os
    from neuprint import Client
    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS", "a3ad6b2c6b4319d0edde6d62215439f1a4b8e7477e457910100ac2450a1bb7e6")
    client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token)
    res = client.fetch_custom("MATCH (n:Neuron) WHERE n.type = 'DNa01' RETURN count(n) as count")
    assert res["count"].iloc[0] > 0, "Expected at least 1 DNa01 neuron in MaleCNS v1.0"
