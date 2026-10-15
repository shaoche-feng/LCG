"""Unlearned channel-concatenated history, owned by one environment loop."""
import torch


class FrameHistory:
    def __init__(self, observation, length, prompt=None):
        self.length = length
        self.channels = observation.shape[1]
        self.state = self.initial(observation, prompt)

    def initial(self, observation, prompt=None):
        """A real reset repeats its first observation. An imagined reset passes its replay
        prompt (frames (b, t, c, h, w), valid (b, t)) and starts from the latest `length`
        valid frames in chronological order, repeating the earliest valid frame when
        fewer are available -- the stack a real episode holds at that point."""
        if prompt is None:
            return observation.repeat(1, self.length, 1, 1)
        frames, valid = prompt
        b, t = valid.shape
        # Replay pads only before an episode's start, so valid frames form a suffix.
        first = t - valid.sum(1, keepdim=True)
        positions = torch.arange(t, device=valid.device)
        if not torch.equal(valid, positions >= first):
            raise ValueError("Replay prompts must pad only before an episode's start")
        index = torch.maximum(torch.arange(t - self.length, t, device=valid.device), first)
        # Advanced indexing copies: the history never aliases the environment's buffers.
        return frames[torch.arange(b, device=valid.device)[:, None], index].flatten(1, 2)

    def successor(self, observation, indices=None):
        state = self.state if indices is None else self.state[indices]
        return torch.cat((state[:, self.channels:], observation), dim=1)

    def advance(self, observation, reset, prompt=None):
        """prompt, if given, holds the replay prompts of the reset rows, in batch order."""
        successor = self.successor(observation)
        if prompt is None:
            # Allocate new storage: previously retained action states never change.
            self.state = torch.where(reset[:, None, None, None], self.initial(observation), successor)
        else:
            successor[reset] = self.initial(observation[reset], prompt)
            self.state = successor
