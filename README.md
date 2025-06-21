# QR life back

<table border="0">
 <tbody>
  <tr>
   <td align="center" valign="center">
    Original<br>
    <img src="examples/original.png" alt="Original" />
   </td>
   <td align="center" valign="center">
     Conway's Game of Life<br>
     &rarr;
   </td>
   <td align="center" valign="center">
    Transformed<br>
    <img src="examples/transformed.png" alt="Transformed" />
   </td>
  </tr>
 </tbody>
</table>

Infer an original QR code (the "Original" above) from a QR code-like grid
representation (the "Transformed" above) which are constructed from the original
QR code by applying the rules of
[Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life).

## How to use

### Setup

1. Install [asdf](https://asdf-vm.com/).
1. Install asdf plugins for [Python](https://www.python.org/) and [uv](https://github.com/astral-sh/uv).

    ```bash
    asdf plugin add python
    asdf plugin add uv
    ```

1. Install tools.

    ```bash
    asdf install
    ```

1. Install Python packages.

    ```bash
    uv sync
    ```

### Usage example

```bash
uv run python qr_life_back.py \
   --input_qrcode_cell_size=21 \
   --input_margin_cell_size=2 \
   --input_original_hint_file_path='examples/original_hint.txt' \
   --input_transformed_file_path='examples/transformed.txt' \
   --output_inferred_original_image_file_path='examples/inferred_original.png'
```
