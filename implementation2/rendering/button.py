from __future__ import annotations

import time
from typing import Callable, Optional, Union, Tuple
import pygame

from implementation2.rendering.color import Color, Colors
from implementation2.rendering.event_handler import EventHandler
from implementation2.rendering.label import Label
from implementation2.rendering.ui_object import UIObject


class Button(UIObject):

    def __init__(self, display: pygame.Surface, position: Tuple[int, int], size: Tuple[int, int], font,
                 center_x: bool = True, center_y: bool = True, text: str = None, img: Union[pygame.Surface, str] = None,
                 color: Optional[Color] = Colors.red, icon: str = None, font_color: Color = Colors.black, bold: bool = False,
                 text_orientation: str = "center", padding: int = 0, border_color: Color = None, border_radius: int = 0,
                 border_width: int = 2, hover_effects: bool = True, hover_color: Color = None, render_with_img: bool = False,
                 id_: int = None, info: str = "") -> None:

        super().__init__(display, position, size, padding, border_color, border_radius, border_width)

        self.display = display
        self.x, self.y = position
        self.size = size
        self.text = text
        self.font = font
        self.img = img
        self.icon = icon
        self.color = color
        self.default_color = color
        self.font_color = font_color
        self.bold = bold
        self.text_orientation = text_orientation
        self.padding = padding
        self.border_color = border_color
        self.border_radius = border_radius
        self.border_width = border_width
        self.center_x = center_x
        self.center_y = center_y
        self.hover_effects = hover_effects
        self.hover_color = hover_color
        self.render_with_img = render_with_img
        self.id_ = id_
        self.info = info

        self.label = None
        if self.text is not None:
            self.label = Label(self.display, (self.x if center_x else self.x + size[0] // 2, self.y), size, font, text=text,
                               font_color=font_color, bold=bold, text_orientation=text_orientation, padding=0)

        if isinstance(size, str) and self.label:
            if size == "fit":
                self.w, self.h = self.label.get_size()
        else:
            self.w, self.h = size

        if self.padding is not None:
            if self.label:
                self.w += max(self.padding - (self.w - self.label.get_size()[0]), 0)
                self.h += max(self.padding - (self.h - self.label.get_size()[1]), 0)
            if self.img:
                self.w += self.padding
                self.h += self.padding

        if not center_x:
            self.x += self.w // 2
        if not center_y:
            self.y += self.h // 2

        if self.img is not None and isinstance(self.img, pygame.Surface):
            self.img = pygame.transform.scale(img, size)

        self.rect = pygame.Rect(self.x - self.w // 2, self.y - self.h // 2, self.w, self.h)

        self.on_hover_listener = None
        self.on_click_listener = None
        self.on_hold_listener = None

        self.hover_class = None
        self.click_class = None
        self.hold_class = None

        self.hover_pass_self = False
        self.click_pass_self = False
        self.hold_pass_self = False

        self.hover_args = ()
        self.click_args = ()
        self.hold_args = ()

        self.is_hovering = False
        self.is_clicked = False
        self.is_held = False
        self.was_held = False

        self.last_hovered = False
        self.last_held = False

        self.disable_time = 0.1
        self.init = False
        self.init_time = 0

        self.icon_padding = 5

    def update_vars(self) -> None:
        self.label = None
        if self.text is not None:
            self.label = Label(self.display, (self.x if self.center_x else self.x + self.size[0] // 2, self.y), self.size,
                               self.font, text=self.text, font_color=self.font_color, bold=self.bold,
                               text_orientation=self.text_orientation, padding=self.padding)

        if isinstance(self.size, str) and self.label:
            if self.size == "fit":
                self.w, self.h = self.label.get_size()
        else:
            self.w, self.h = self.size

        if self.padding is not None:
            if self.label:
                self.w += max(self.padding - (self.w - self.label.get_size()[0]), 0)
                self.h += max(self.padding - (self.h - self.label.get_size()[1]), 0)
            if self.img:
                self.w += self.padding
                self.h += self.padding

        if not self.center_x:
            self.x += self.w // 2
        if not self.center_y:
            self.y += self.h // 2

        if self.img is not None and isinstance(self.img, pygame.Surface):
            self.img = pygame.transform.scale(self.img, self.size)

        self.rect = pygame.Rect(self.x - self.w // 2, self.y - self.h // 2, self.w, self.h)

    def reinit(self) -> None:
        self.disable_time = 0.1
        self.init_time = 0
        self.init = False

    def set_text(self, text: str) -> None:
        self.text = text
        self.update_vars()
        self.label.set_text(text)

    def update(self, event_handler: EventHandler) -> None:
        if not self.init:
            self.init = True
            self.init_time = time.time()
            self.is_hovering = False

        if self.color and self.hover_effects:
            if self.last_hovered and not self.is_hovering:  # e.i. on_exit()
                self.color = self.default_color
            elif not self.last_hovered and self.is_hovering:  # e.i. on_enter()
                if self.hover_color:
                    self.color = self.hover_color
                else:
                    self.color = (max(self.color[0] - 50, 0), max(self.color[1] - 50, 0), max(self.color[2] - 50, 0))

        self.last_hovered = self.is_hovering
        self.last_held = self.is_held

        self.is_clicked = False
        self.is_hovering = False
        self.is_held = False

        if time.time() - self.init_time <= self.disable_time:
            return

        pos = event_handler.get_pos()

        if self.rect.collidepoint(pos):
            if event_handler.presses["left"]:
                self.was_held = True

            if event_handler.releases["left"] and self.was_held:
                self.on_click(*pos)
                self.is_clicked = True
                if callable(self.on_click_listener):
                    if self.click_pass_self:
                        self.on_click_listener(self.click_class, *pos, self, *self.click_args)
                    else:
                        self.on_click_listener(self.click_class, *pos, *self.click_args)
            elif event_handler.held["left"]:
                self.on_hold(*pos)
                self.is_held = True
                if callable(self.on_hold_listener):
                    if self.hold_pass_self:
                        self.on_hold_listener(self.hold_class, *pos, self, *self.hold_args)
                    else:
                        self.on_hold_listener(self.hold_class, *pos, *self.hold_args)

            self.on_hover(*pos)
            self.is_hovering = True
            if callable(self.on_hover_listener):
                if self.hover_pass_self:
                    self.on_hover_listener(self.hover_class, *pos, self, *self.hover_args)
                else:
                    self.on_hover_listener(self.hover_class, *pos, *self.hover_args)
        else:
            self.was_held = False

    def render(self) -> None:
        if self.img is None:
            self.render_non_image()
        else:
            if self.render_with_img:
                self.render_non_image()
            self.render_image()

        if self.label:
            self.label.render()

    def render_non_image(self) -> None:
        if self.color is None and self.icon is None:
            return

        surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)

        pygame.draw.rect(surf, self.color, [0, 0, self.rect.w, self.rect.h], width=0,
                         border_radius=self.border_radius)
        if self.border_color:
            pygame.draw.rect(surf, self.border_color, [0, 0, self.rect.w, self.rect.h],
                             width=self.border_width, border_radius=self.border_radius)
        self.display.blit(surf, (self.rect.x, self.rect.y))

        if self.icon == "+":
            w = min(self.rect.width - self.icon_padding * 2, self.rect.height - self.icon_padding * 2)
            pygame.draw.line(self.display, self.border_color, (self.rect.centerx, self.rect.centery - w // 2 + self.icon_padding),
                             (self.rect.centerx, self.rect.centery + w // 2 - self.icon_padding))
            pygame.draw.line(self.display, self.border_color, (self.rect.centerx - w // 2 + self.icon_padding, self.rect.centery),
                             (self.rect.centerx + w // 2 - self.icon_padding, self.rect.centery))
        elif self.icon == "x":
            pygame.draw.line(self.display, self.border_color, self.rect.topleft, self.rect.bottomright,
                             self.border_width)
            pygame.draw.line(self.display, self.border_color, self.rect.topright, self.rect.bottomleft,
                             self.border_width)

    def render_image(self) -> None:
        if isinstance(self.img, str):
            if self.img == "x":
                pygame.draw.line(self.display, self.border_color, self.rect.topleft, self.rect.bottomright, self.border_width)
                pygame.draw.line(self.display, self.border_color, self.rect.topright, self.rect.bottomleft, self.border_width)
            if self.img == "o":
                pygame.draw.circle(self.display, self.border_color, self.rect.center, self.rect.w // 2)
        else:
            self.display.blit(self.img, (self.rect.x + self.padding // 2, self.rect.y + self.padding // 2))

    def on_hover(self, x: int, y: int) -> None:
        pass

    def on_click(self, x: int, y: int) -> None:
        pass

    def on_hold(self, x: int, y: int) -> None:
        pass

    # Getters and Setters

    def set_on_hover_listener(self, listener: Callable, cls, pass_self: bool = False, args=()) -> Button:
        self.on_hover_listener = listener
        self.hover_class = cls
        self.hover_pass_self = pass_self
        self.hover_args = args
        return self

    def set_on_click_listener(self, listener: Callable, cls, pass_self: bool = False, args=()) -> Button:
        self.on_click_listener = listener
        self.click_class = cls
        self.click_pass_self = pass_self
        self.click_args = args
        return self

    def set_on_hold_listener(self, listener: Callable, cls, pass_self: bool = False, args=()) -> Button:
        self.on_hold_listener = listener
        self.hold_class = cls
        self.hold_pass_self = pass_self
        self.hold_args = args
        return self

    def move(self, x: int = None, y: int = None, cx: bool = True, cy: bool = True) -> None:
        if x is not None:
            self.x = x
            if not cx:
                self.x += self.w // 2
        if y is not None:
            self.y = y
            if not cy:
                self.y += self.h // 2

        self.rect.x = self.x
        self.rect.y = self.y

        self.update_vars()

    def get_text(self) -> str:
        if self.label is None:
            return ""
        return self.label.text

    def get_size(self) -> Tuple[int, int]:
        return self.rect.size

    def get_center(self) -> Tuple[int, int]:
        return self.rect.center

    def set_size(self, size: Tuple[int, int]) -> None:
        self.size = size

    def set_display(self, surface: pygame.Surface) -> None:
        self.display = surface
