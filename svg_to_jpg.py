from wand.image import Image
from wand.color import Color


def convert_svg_to_jpeg(svg_path, jpeg_path, dpi=300):
    # Open the SVG file using Wand
    with Image(filename=svg_path, resolution=dpi) as img:
        # Set the background color to white for JPEG conversion
        # Since JPEG does not support transparency
        with Image(width=img.width, height=img.height, background=Color('white')) as bg:
            bg.composite(img, 0, 0)
            bg.format = 'jpeg'
            bg.save(filename=jpeg_path)


# Specify the SVG and output JPEG path
svg_file = r"C:\Users\wangs\My Drive\Dissertation\finite element project\DLArch.svg"
jpeg_file = r"C:\Users\wangs\My Drive\Dissertation\finite element project\Fig1 Deep learning.jpg"

# Convert and save the JPEG
convert_svg_to_jpeg(svg_file, jpeg_file, dpi=2000)
