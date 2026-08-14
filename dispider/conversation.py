"""Qwen prompt assembly used by Dispider inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SeparatorStyle(Enum):
    PHI = "qwen"


@dataclass
class Conversation:
    system: str
    roles: tuple[str, str]
    messages: list[list[Optional[str]]] = field(default_factory=list)
    sep: str = " "
    sep2: str = "<|im_end|>"
    sep_style: SeparatorStyle = SeparatorStyle.PHI

    def get_prompt(self) -> str:
        prompt = self.system + self.sep
        separators = (self.sep, self.sep2)
        for index, (role, message) in enumerate(self.messages):
            if message is None:
                prompt += role + ":"
            else:
                prompt += role + ": " + message + separators[index % 2]
        return prompt

    def append_message(self, role: str, message: Optional[str]) -> None:
        self.messages.append([role, message])

    def copy(self) -> "Conversation":
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[message.copy() for message in self.messages],
            sep=self.sep,
            sep2=self.sep2,
            sep_style=self.sep_style,
        )


conv_qwen = Conversation(
    system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's "
        "questions."
    ),
    roles=("USER", "ASSISTANT"),
)

default_conversation = conv_qwen
conv_templates = {"qwen": conv_qwen}

__all__ = [
    "Conversation",
    "SeparatorStyle",
    "conv_qwen",
    "conv_templates",
    "default_conversation",
]
