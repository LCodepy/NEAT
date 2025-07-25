from abc import ABC, abstractmethod
from typing import Tuple

import pygame

from implementation2.rendering.event_handler import EventHandler


class UIObject(ABC):

    def __init__(self, display: pygame.Surface, position: Tuple[int, int], size: Tuple[int, int],
                 padding: int, border_color: Tuple[int, int, int] = None, border_radius: int = 0, border_width: int = 0) -> None:
        self.display = display
        self.x, self.y = position
        self.w, self.h = size
        self.padding = padding
        self.border_color = border_color
        self.border_radius = border_radius
        self.border_width = border_width

    @abstractmethod
    def update(self, event_handler: EventHandler) -> None:
        """Updates the UI object."""

    @abstractmethod
    def render(self) -> None:
        """Renders the UI object."""

    @abstractmethod
    def update_vars(self) -> None:
        """Reinit the UI object."""

    @abstractmethod
    def move(self, x: int = None, y: int = None, cx: bool = True, cy: bool = True) -> None:
        """Changes the UI object position."""

    @abstractmethod
    def get_size(self) -> Tuple[int, int]:
        """Returns the size of the UI object."""

    @abstractmethod
    def get_center(self) -> Tuple[int, int]:
        """Returns the center of the UI object."""

    @abstractmethod
    def set_size(self, size: Tuple[int, int]) -> None:
        """Sets the new size fot the UI object."""

    @abstractmethod
    def set_display(self, surface: pygame.Surface) -> None:
        """Sets new display surface for the UI object."""
