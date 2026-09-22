"""Unlearned channel-concatenated history, owned by one environment loop."""
import torch


class FrameHistory:
    def __init__(self, observation, length):
        self.length = length
        self.channels = observation.shape[1]
        self.state = self.initial(observation)

    def initial(self, observation):
        return observation.repeat(1, self.length, 1, 1)

    def successor(self, observation, indices=None):
        state = self.state if indices is None else self.state[indices]
        return torch.cat((state[:, self.channels:], observation), dim=1)

    def advance(self, observation, reset):
        successor = self.successor(observation)
        # Allocate new storage: previously retained action states never change.
        self.state = torch.where(reset[:, None, None, None], self.initial(observation), successor)
