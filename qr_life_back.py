"""Infer the original cell states from a QR code-like grid representation.

Infer the original cell states of a QR code from a transformed cell states
which are constructed from the original cell states by applying the rules of
Conway's Game of Life.
"""

import argparse
import enum
import pathlib
from typing import Self

import cv2
import numpy


class CellState(enum.Enum):
    """Cell state enumeration."""

    UNKNOWN = "?"
    LIVE = "X"
    DEAD = " "


class CellBuffer:
    """Buffer for storing cell states in a grid."""

    def __init__(self, buffer_size: int) -> None:
        """Initialize the cell buffer with a given size.

        Args:
            buffer_size (int): The size of the buffer (number of cells in each dimension).
        """
        if buffer_size <= 0:
            raise ValueError("Buffer size must be a positive integer.")
        self._buffer_size = buffer_size
        self._cell_states = [
            [CellState.UNKNOWN] * buffer_size for _ in range(buffer_size)
        ]

    @property
    def buffer_size(self) -> int:
        """Get the size of the buffer."""
        return self._buffer_size

    def get_cell_state(self, i: int, j: int) -> CellState:
        """Get the state of the cell at position (i, j)."""
        if i < 0 or self.buffer_size <= i:
            raise ValueError("The row index is out of bounds.")
        if j < 0 or self.buffer_size <= j:
            raise ValueError("The column index is out of bounds.")
        return self._cell_states[i][j]

    def set_cell_state(self, i: int, j: int, new_cell_state: CellState) -> None:
        """Set the state of the cell at position (i, j) to new_cell_state."""
        if i < 0 or self.buffer_size <= i:
            raise ValueError("The row index is out of bounds.")
        if j < 0 or self.buffer_size <= j:
            raise ValueError("The column index is out of bounds.")
        self._cell_states[i][j] = new_cell_state

    def get_next_cell_state(self, i: int, j: int) -> CellState:
        """Get the next state of the cell at position (i, j).

        Quoted from https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
        > 1. Any live cell with fewer than two live neighbours dies, as if by
        >    underpopulation.
        > 2. Any live cell with two or three live neighbours lives on to the next
        >    generation.
        > 3. Any live cell with more than three live neighbours dies, as if by
        >    overpopulation.
        > 4. Any dead cell with exactly three live neighbours becomes a live cell,
        >    as if by reproduction.
        """
        current_cell_state = self.get_cell_state(i, j)
        if current_cell_state is CellState.UNKNOWN:
            raise ValueError("The cell state is unknown.")
        live_cell_state_count = 0
        for neighbor_i, neighbor_j in (
            (i - 1, j - 1),
            (i - 1, j),
            (i - 1, j + 1),
            (i, j - 1),
            (i, j + 1),
            (i + 1, j - 1),
            (i + 1, j),
            (i + 1, j + 1),
        ):
            cell_state = self.get_cell_state(neighbor_i, neighbor_j)
            if cell_state is CellState.LIVE:
                live_cell_state_count += 1
            elif cell_state is not CellState.DEAD:
                raise ValueError(f"The cell state is unexpected: {cell_state}")
        if current_cell_state is CellState.LIVE:
            if live_cell_state_count < 2:
                # Underpopulation.
                return CellState.DEAD
            elif 3 < live_cell_state_count:
                # Overpopulation.
                return CellState.DEAD
            else:
                # Next generation.
                return CellState.LIVE
        else:
            if live_cell_state_count == 3:
                # Reproduction.
                return CellState.LIVE
            else:
                return CellState.DEAD

    @classmethod
    def from_text(cls, text_cell_state: str, buffer_size: int) -> Self:
        """Create a CellBuffer from a text representation of cell states.

        LIVE cells are represented by 'X', DEAD cells by ' ', and UNKNOWN cells by '?'.
        """
        cell_buffer = cls(buffer_size)
        rows = text_cell_state.splitlines(keepends=False)
        if len(rows) != buffer_size:
            raise ValueError("The number of rows does not match buffer size.")
        for i, row in enumerate(rows):
            if len(row) != buffer_size:
                raise ValueError("The number of columns does not match buffer size.")
            for j, col in enumerate(row):
                cell_state = CellState(col)
                cell_buffer.set_cell_state(i, j, cell_state)
        return cell_buffer

    def to_text(self) -> str:
        """Convert the cell buffer to a text representation of cell states.

        LIVE cells are represented by 'X', DEAD cells by ' ', and UNKNOWN cells by '?'.
        """
        return "\n".join(
            "".join(self.get_cell_state(i, j).value for j in range(self.buffer_size))
            for i in range(self.buffer_size)
        )

    def to_image(self) -> numpy.typing.NDArray[numpy.uint8]:
        """Convert the cell buffer to an image representation."""
        # Because the cell buffer is too small to visualize as an image, we need to
        # scale it up to visualize it.
        pixels_per_cell = 2
        image = numpy.zeros(
            (
                self.buffer_size * pixels_per_cell,
                self.buffer_size * pixels_per_cell,
            ),
            numpy.uint8,
        )
        for i in range(self.buffer_size):
            for j in range(self.buffer_size):
                cell_state = self.get_cell_state(i, j)
                if cell_state is CellState.DEAD:
                    for image_i in range(
                        i * pixels_per_cell, (i + 1) * pixels_per_cell
                    ):
                        for image_j in range(
                            j * pixels_per_cell, (j + 1) * pixels_per_cell
                        ):
                            image[image_i, image_j] = 0xFF
                elif cell_state is not CellState.LIVE:
                    raise ValueError(f"The cell state is unexpected: {cell_state}")
        return image

    def save_as_image(self, output_file_path: pathlib.Path) -> None:
        """Save the cell buffer as an image file."""
        # Create the parent directory if it does not exist because cv2.imwrite
        # does not create it.
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file_path), self.to_image())


def _is_valid_qr_code(cell_buffer: CellBuffer) -> bool:
    """Check if the cell buffer contains a valid QR code."""
    detector = cv2.QRCodeDetector()
    image = cell_buffer.to_image()
    straight_qrcode = detector.detectAndDecode(image)[2]
    return straight_qrcode is not None


def _infer_original_cell_states(
    original_cell_buffer: CellBuffer,
    transformed_cell_buffer: CellBuffer,
    i: int,
    j: int,
    margin_cell_size: int,
) -> bool:
    """Recursively infer the original cell states to match the transformed cell buffer."""
    if (j + 1) >= transformed_cell_buffer.buffer_size:
        if (i + 1) >= transformed_cell_buffer.buffer_size:
            # All cells have been processed.
            return _is_valid_qr_code(original_cell_buffer)
        next_i = i + 1
        next_j = margin_cell_size
    else:
        next_i = i
        next_j = j + 1
    original_cell_state = original_cell_buffer.get_cell_state(i, j)
    # If the original cell state is UNKNOWN, try both LIVE and DEAD states.
    if original_cell_state is CellState.UNKNOWN:
        candidate_cell_states: tuple[CellState, ...] = (CellState.LIVE, CellState.DEAD)
    else:
        candidate_cell_states = (original_cell_state,)
    transformed_cell_state = transformed_cell_buffer.get_cell_state(i - 1, j - 1)
    for candidate_cell_state in candidate_cell_states:
        original_cell_buffer.set_cell_state(i, j, candidate_cell_state)
        if original_cell_buffer.get_next_cell_state(
            i - 1, j - 1
        ) is transformed_cell_state and _infer_original_cell_states(
            original_cell_buffer,
            transformed_cell_buffer,
            next_i,
            next_j,
            margin_cell_size,
        ):
            return True
    # If no valid state was found, restore the original state.
    original_cell_buffer.set_cell_state(i, j, original_cell_state)
    return False


def infer_original(
    input_qrcode_cell_size: int,
    input_margin_cell_size: int,
    input_original_hint_file_path: pathlib.Path,
    input_transformed_file_path: pathlib.Path,
) -> CellBuffer:
    """Infer the original cell states from the transformed cell buffer."""
    # The input original and transformed files are expected to be text files
    # containing the cell states in a grid format with surrounding margins.
    buffer_size = input_qrcode_cell_size + 2 * input_margin_cell_size
    original_cell_buffer = CellBuffer.from_text(
        input_original_hint_file_path.read_text(), buffer_size
    )
    transformed_cell_buffer = CellBuffer.from_text(
        input_transformed_file_path.read_text(), buffer_size
    )
    if not _infer_original_cell_states(
        original_cell_buffer,
        transformed_cell_buffer,
        input_margin_cell_size,
        input_margin_cell_size,
        input_margin_cell_size,
    ):
        raise ValueError("Failed to infer the original cell states.")
    return original_cell_buffer


def main() -> None:
    """Main function to parse arguments and infer the original cell states."""
    parser = argparse.ArgumentParser(description="Infer original cell states.")
    parser.add_argument(
        "--input_qrcode_cell_size",
        required=True,
        type=int,
        help="QR code cell size of the original and transformed QR codes.",
    )
    parser.add_argument(
        "--input_margin_cell_size",
        required=True,
        type=int,
        help="Margin cell size around the QR code.",
    )
    parser.add_argument(
        "--input_original_hint_file_path",
        required=True,
        type=pathlib.Path,
        help="Path to the input original hint cell states file.",
    )
    parser.add_argument(
        "--input_transformed_file_path",
        required=True,
        type=pathlib.Path,
        help="Path to the input transformed cell states file.",
    )
    parser.add_argument(
        "--output_inferred_original_image_file_path",
        required=False,
        type=pathlib.Path,
        help="Optional path to save the inferred original cell states image.",
    )
    args = parser.parse_args()

    inferred_original_cell_buffer = infer_original(
        args.input_qrcode_cell_size,
        args.input_margin_cell_size,
        args.input_original_hint_file_path,
        args.input_transformed_file_path,
    )
    print(inferred_original_cell_buffer.to_text())
    if args.output_inferred_original_image_file_path:
        inferred_original_cell_buffer.save_as_image(
            args.output_inferred_original_image_file_path
        )


if __name__ == "__main__":
    main()
