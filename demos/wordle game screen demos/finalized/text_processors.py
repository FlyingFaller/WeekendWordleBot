"""
Defines a modular system for processing and rendering text within the game tiles.
"""
import pyfiglet

class TextProcessor:
    """Base class for all text processors."""
    def process(self, letter: str, size: tuple[int, int], state: 'LetterState') -> str:
        """
        Processes a single letter for display. This base method is a fallback.
        
        Args:
            letter: The character to process.
            size: The (width, height) of the content area.
            state: The current state of the LetterSquare.

        Returns:
            The processed string to be rendered.
        """
        return letter

class FigletProcessor(TextProcessor):
    """A text processor that renders letters using pyfiglet."""

    def _render_filled(self, letter: str) -> str:
        """
        Helper to render text for the 'filled' state.
        This logic can be modified to be driven by external settings.
        """
        # For now, this uses the "bigfig" font and uppercase letters.
        font_name = "calvin_s"
        try:
            renderer = pyfiglet.Figlet(font=font_name)
            return renderer.renderText(letter.upper()).rstrip('\n')
        except pyfiglet.FontNotFound:
            return letter.upper()

    def _render_recommendation(self, letter: str) -> str:
        """
        Helper to render text for the 'recommendation' state.
        This logic can be modified to be driven by external settings.
        """
        # For now, this uses the "slant" font.
        font_name = "calvin_s"
        try:
            renderer = pyfiglet.Figlet(font=font_name)
            return renderer.renderText(letter.lower()).rstrip('\n')
        except pyfiglet.FontNotFound:
            return letter.upper()

    def process(self, letter: str, size: tuple[int, int], state: 'LetterState') -> str:
        """
        Conditionally renders either ASCII art or a single character.
        """
        # 1. Fallback to normal text if the tile is too small
        if size[1] < 5:
            return letter

        # 2. Dispatch to the correct rendering helper based on state
        if state.name == 'FILLED':
            return self._render_filled(letter)
        else:  # RECOMMENDATION or EMPTY
            return self._render_recommendation(letter)
