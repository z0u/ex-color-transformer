# type: ignore

from __future__ import annotations

from typing import Any, Optional, Self, overload, Literal, TypeAlias, Union

import torch


TF: TypeAlias = Literal['step-end', 'ease', 'linear', 'ease-in', 'ease-out', 'ease-in-out']


class TrainingTimeline:
    def set(self, name: str, value: str | int | float) -> Self:
        ...

    def advance(self) -> int:
        pass


class KeyframeBuilder:
    """Helper class for fluent API to build keyframes."""

    def __init__(self, timeline, step):
        self.timeline = timeline
        self.step = step

    def key(self, param_name: str, value) -> Self:
        """Set parameter value at this keyframe."""
        # Handle nested parameters with dot notation
        pass

    def add_event(self, event_type, params=None) -> Self:
        """Add an event at this keyframe."""
        pass


t = TrainingTimeline()
t.log('Initializing parameters')

# These are just hyperparameters - the actual functions are defined in code

t.set('learning_rate', 0)

# Training data comes from a HSV cube. These parameters determine how the color space is sampled.
# Hue is always sampled over the full range.
t.set('dataset.hsv.sampler', 'regular')  # Start with regular intervals
t.set('dataset.hsv.samples', 6)  # Number of samples to draw per epoch
t.set('dataset.hsv.s.low', 1.0)  # Lower bound for saturation. Upper bound is always 1.0
t.set('dataset.hsv.v.low', 1.0)  # Lower bound for brightness. Upper bound is always 1.0

# Validation data comes from an RGB cube.
t.set('dataset.rgb.r.steps', 5)
t.set('dataset.rgb.g.steps', 5)
t.set('dataset.rgb.b.steps', 5)

# Loss terms
t.set('loss.recon.w', 1.0)     # Reconstruction loss (primary objective)

# Regularization loss terms (relating to latent space)
t.set('reg.unitary.w', 0.1)   # Vectors of length 1
t.set('reg.smooth.w', 0.1)    # Smoothness of the latent space
t.set('reg.planar.w', 0.1)    # Planarity of the latent space
t.set('reg.separate.w', 0.1)  # Separation of points in the latent space
t.set('reg.anchor.w', 0)      # Distance from anchored data points


## Phase 1: Fitting primary colors

t.log('Fitting primary colors')

# `transition` sets a target value, interpolation will happen over subsequent steps
# Ints can be smoothly interpolated, but will be rounded to the nearest integer
t.transition('learning_rate', 0.01, duration=100)
t.transition('reg.separate.w', 0.0, duration=1000)

# Wait for a metric to reach a target value or for a timeout
# 'loss' is the combination of all regularization terms
t.wait_for((M('loss') < 1e-6) | Steps(1000))
t.log('Basis found; storing as anchor')
t.create_anchor('primary_colors', latents=['encoder.2.w'])
t.set('reg.anchor.target', 'primary_colors')

t.log('Adding variations in hue')
t.set('dataset.hsv.sampler', 'stochastic')
t.transition('dataset.hsv.samples', 360, duration=1000)
t.transition('reg.anchor.w', 0.1, duration=1000)
t.transition('reg.planarity.w', 0.0, duration=1000)

# Wait for at least 1000 steps, and then continue waiting until the loss is below 1e-4 (or 1000 more steps)
t.wait_for(1000)
t.wait_for((M('loss') < 1e-4) | Steps(1000))

t.log('Adding variations in brightness and saturation')
t.transition('dataset.train.h_sampler.steps', 60, duration=1000)
t.transition('learning_rate', 0.001, duration=1000)

t.wait_for((M('')))
