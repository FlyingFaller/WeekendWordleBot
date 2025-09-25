# progress_bar.py

from __future__ import annotations
from typing import Optional

from rich.style import Style
from textual.app import ComposeResult, RenderResult
from textual.color import Gradient
from textual.geometry import clamp
from textual.reactive import reactive
from textual.renderables.bar import Bar as BarRenderable
from textual.widgets import Label, ProgressBar

# --- Import internal Textual classes for subclassing ---
from textual.widgets._progress_bar import (
    Bar,
    ETAStatus,
    PercentageStatus,
    UNUSED,
    UnusedParameter,
)

class TimeElapsedStatus(Label):
    """A label to display the elapsed time of the progress bar."""

    DEFAULT_CSS = """
    TimeElapsedStatus {
        width: 9;
        content-align-horizontal: left;
        margin-right: 1; /* Adds a nice space between the time and the bar */
    }
    """
    time_elapsed: reactive[float | None] = reactive[Optional[float]](None)
    """Elapsed number of seconds, or `None` if not started."""

    def render(self) -> RenderResult:
        """Render the elapsed time display."""
        time_elapsed = self.time_elapsed
        if time_elapsed is None:
            return "00:00:00"
        else:
            minutes, seconds = divmod(round(time_elapsed), 60)
            hours, minutes = divmod(minutes, 60)
            return f"{hours:02}:{minutes:02}:{seconds:02}"


class PatchedBar(Bar):
    """
    Inherited Bar with patches for indeterminate animation and 'bar--complete' styling.
    """
    def render(self) -> RenderResult:
        if self.percentage is None:
            return self.render_indeterminate()
        else:
            is_complete = self.percentage >= 1
            bar_style = (
                self.get_component_rich_style("bar--complete")
                if is_complete
                else self.get_component_rich_style("bar--bar")
            )
            render_gradient = None if is_complete else self.gradient
            return BarRenderable(
                highlight_range=(0, self.size.width * self.percentage),
                highlight_style=Style.from_color(bar_style.color),
                background_style=Style.from_color(bar_style.bgcolor),
                gradient=render_gradient,
            )

    def render_indeterminate(self) -> RenderResult:
        width = self.size.width
        if not width:
            return ""
        highlighted_bar_width = 0.25 * width
        total_imaginary_width = width + highlighted_bar_width
        start: float
        end: float
        if self.app.animation_level == "none":
            start, end = 0, width
        else:
            speed = 30
            start = (speed * self._clock.time) % (2 * total_imaginary_width)
            if start > total_imaginary_width:
                start = 2 * total_imaginary_width - start
            start -= highlighted_bar_width
            end = start + highlighted_bar_width
        bar_style = self.get_component_rich_style("bar--indeterminate")
        return BarRenderable(
            highlight_range=(max(0, start), min(end, width)),
            highlight_style=Style.from_color(bar_style.color),
            background_style=Style.from_color(bar_style.bgcolor),
        )

class PatchedProgressBar(ProgressBar):
    """The fully-patched ProgressBar that includes the TimeElapsedStatus."""

    _display_time_elapsed: reactive[float | None] = reactive[Optional[float]](None)

    def __init__(
        self,
        total: float | None = None,
        *,
        show_bar: bool = True,
        show_percentage: bool = True,
        show_eta: bool = True,
        show_time_elapsed: bool = True, # New parameter
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        gradient: Gradient | None = None,
    ):
        super().__init__(
            total=total,
            show_bar=show_bar,
            show_percentage=show_percentage,
            show_eta=show_eta,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
            gradient=gradient,
        )
        self.show_time_elapsed = show_time_elapsed

    def _compute_percentage(self) -> float | None:
        """
        Override the percentage computation to show an indeterminate state
        when progress is zero or the total is unknown/zero.
        """
        if self.progress == 0:
            return None
        if not self.total:
            return None
        return clamp(self.progress / self.total, 0.0, 1.0)

    def _reset_eta(self) -> None:
        """Fix bug in ETA object that corrupts data on reset"""
        self._eta._samples = [(0.0, 0.0)]
        self._eta._add_count = 0

    def watch_percentage(self, old_percentage: float | None, new_percentage: float | None) -> None:
        """When the state changes from indeterminate to determinate, reset the clock."""
        if old_percentage is None and new_percentage is not None:
            self._clock.reset()
            self._reset_eta()

    def compose(self) -> ComposeResult:
        """Creates the child widgets, adding the elapsed time at the start."""
        if self.show_time_elapsed:
            yield TimeElapsedStatus(id="elapsed").data_bind(
                time_elapsed=PatchedProgressBar._display_time_elapsed
            )
        if self.show_bar:
            yield PatchedBar(id="bar", clock=self._clock).data_bind(
                ProgressBar.percentage, ProgressBar.gradient
            )
        if self.show_percentage:
            yield PercentageStatus(id="percentage").data_bind(ProgressBar.percentage)
        if self.show_eta:
            yield ETAStatus(id="eta").data_bind(eta=ProgressBar._display_eta)

    def update(
        self,
        *,
        total: None | float | UnusedParameter = UNUSED,
        progress: float | UnusedParameter = UNUSED,
        advance: float | UnusedParameter = UNUSED,
    ) -> None:
        """Overrides the original update method to also update elapsed time."""
        current_time = self._clock.time
        if not isinstance(total, UnusedParameter):
            if total is None or total != self.total:
                self._reset_eta()
            self.total = total
        def add_sample() -> None:
            if self.progress is not None and self.total:
                self._eta.add_sample(current_time, self.progress / self.total)
        if not isinstance(progress, UnusedParameter):
            self.progress = progress
            add_sample()
        if not isinstance(advance, UnusedParameter):
            self.progress += advance
            add_sample()

        # Using self.percentage instead of self.total to better align with definition of indeterminate
        self._display_eta = (
            None if self.percentage is None else self._eta.get_eta(current_time)
        )
        if self.percentage is None: # If we are indeterminate
            self._display_time_elapsed = 0.0
        elif self.percentage < 1: # If we aren't indeterminate but we aren't complete either
            self._display_time_elapsed = self._clock.time