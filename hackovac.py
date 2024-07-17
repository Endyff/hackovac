import streamlit as st
import cv2
import numpy as np
import fpdf
import shutil

# Constants
GRID_NORMAL_LINE_SIZE = 1
GRID_NORMAL_LINE_COLOR_BGR = (80, 80, 80)
GRID_BOLD_LINE_EVERY = 5
GRID_BOLD_LINE_SIZE = 2
GRID_BOLD_LINE_COLOR_BGR = (100, 20, 20)

MAX_HEIGHT = 1000
# User settings
width_size = st.sidebar.slider('Šířka výstupu', 10, 500, 200)
threshold = st.sidebar.slider('Práh', 0, 255, 128)
blur_kernel_size = st.sidebar.slider('Rozmazání', 0, 120, 0) 
show_original_image = st.sidebar.checkbox('Zobrazit originál', False)
show_grid = st.sidebar.checkbox('Zobrazit mřížku', False)
invert = st.sidebar.checkbox('Invertovat barvy', False)

# Load the image
image = cv2.imread('examples/cat.jpg')
image_file = st.file_uploader('Výběr souboru', type=['jpg', 'jpeg', 'png', 'webp'])
if image_file is not None:
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

if show_original_image:
    st.image(image)
    st.write(f'Rozměry: {image.shape[1]} x {image.shape[0]}')

image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

if blur_kernel_size: # Kernel size must be odd (or figure out how gaussian_kernel_size can add padding)
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

h,w = image.shape
height_size = int(width_size * h / w)

w_ratio = w // width_size
h_ratio = h // height_size

downscale_image = cv2.resize(image, (width_size, height_size))
d_h, d_w = downscale_image.shape

image = cv2.resize(downscale_image, (width_size * w_ratio, height_size * h_ratio), interpolation=cv2.INTER_NEAREST)
_, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)


# Add grid to the image based on the downscaled image
if invert:
    image = cv2.bitwise_not(image)
    downscale_image = cv2.bitwise_not(downscale_image)

image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Draw grid 
if show_grid:
    # Calculate the size of each grid cell
    cell_width = image.shape[1] // downscale_image.shape[1]
    cell_height = image.shape[0] // downscale_image.shape[0]

    # Draw vertical grid lines
    for x in range(0, downscale_image.shape[1]+1):
        if x % GRID_BOLD_LINE_EVERY:
            line_width = GRID_NORMAL_LINE_SIZE
            line_color = GRID_NORMAL_LINE_COLOR_BGR
        else:
            line_width = GRID_BOLD_LINE_SIZE
            line_color = GRID_BOLD_LINE_COLOR_BGR
        x = x * cell_width
        cv2.line(image, (x, 0), (x, image.shape[0]), line_color, line_width)

    # Draw horizontal grid lines
    for y in range(0, downscale_image.shape[0]+1):
        y = y * cell_height
        if y % GRID_BOLD_LINE_EVERY:
            line_width = GRID_NORMAL_LINE_SIZE
            line_color = GRID_NORMAL_LINE_COLOR_BGR
        else:
            line_width = GRID_BOLD_LINE_SIZE
            line_color = GRID_BOLD_LINE_COLOR_BGR
        cv2.line(image, (0, y), (image.shape[1], y), line_color, line_width)

# Display the binary image
st.image(image)
st.write(f'Rozměry: {width_size} x {height_size}')

num_pages = image.shape[0] // MAX_HEIGHT + 1
# # TODO: Figure out what is good output strategy#
if st.sidebar.button('Stáhnout vygenerovanou předlohu'):
    pdf = fpdf.FPDF('L', 'pt', 'A4')
    for image_part in range(num_pages):
        subimage = image[image_part*MAX_HEIGHT:(image_part+1)*MAX_HEIGHT]
        h, w, _ = subimage.shape
        cv2.imwrite(f'output_image_{image_part}.jpg', cv2.cvtColor(subimage, cv2.COLOR_RGB2BGR))
        pdf.add_page()
        print(pdf.h)
        print(pdf.w)
        height = (h/1000*pdf.h - 100) 
        # width = 
        pdf.image(f'output_image_{image_part}.jpg', x=50, y=50, w=pdf.w-100, h=height)
    pdf.output('output.pdf', 'F')

# Hide deploy button
st.markdown('<style>.stDeployButton {visibility: hidden}</style>', unsafe_allow_html=True)
